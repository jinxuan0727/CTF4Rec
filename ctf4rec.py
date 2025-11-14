import logging
from collections import defaultdict
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import GMA_TransformerEncoder, CLLayer, FDFEncoder, TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.fft as fft
import math

import dgl
from dgl.nn.pytorch import GraphConv

def cal_kl(target, input):
    ### log with sigmoid
    target = torch.sigmoid(target)
    input = torch.sigmoid(input)
    target = torch.log(target + 1e-8)
    input = torch.log(input + 1e-8)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)

def cal_kl_1(target, input):
    target[target<1e-8] = 1e-8
    target = torch.log(target + 1e-8)
    input = torch.log_softmax(input + 1e-8, dim=0)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)
def graph_dual_neighbor_readout(g: dgl.DGLGraph, aug_g: dgl.DGLGraph, node_ids, features):
    _, all_neighbors = g.out_edges(node_ids)
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)
    for_nbr_num = aug_g.out_degrees(node_ids)
    all_neighbors = [set(t.tolist())
                     for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist())
                         for t in foreign_neighbors.split(for_nbr_num.tolist())]
    # sample foreign neighbors
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)
    civil_neighbors = [all_neighbors[i]-foreign_neighbors[i]
                       for i in range(len(all_neighbors))]
    # sample civil neighbors
    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)
    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t)
                           for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()
    foreign_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in foreign_neighbors])
    civil_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]
    # insert zero vector for zero-length neighbors
    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)


def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):
    # Augment the graph with the item sequence, deleting co-occurrence edges in the batched sequences
    # generating indicies like: [1,2] [2,3] ... as the co-occurrence rel.
    # indexing edge data using node indicies and delete them
    # for edge weights, delete them from the raw data using indexed edges
    user_ids = user_ids.cpu().numpy()
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)
    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # Firstly mask selected edge values, returns the true values along with the masked graph.
    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.7):
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=False)

    def forward(self, graph, feature):
        graph = dgl.add_self_loop(graph)
        origin_w, graph = graph_dropout(graph, 1-self.dropout_prob)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        # recover edge weight
        graph.edata['w'] = origin_w
        return final_emb


class CTF4Rec(SequentialRecommender):

    def __init__(self, config, dataset, external_data):
        super(CTF4Rec, self).__init__(config, dataset)

        self.config = config
        self.device = config["device"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # CLF4Rec
        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.lmd_tf = config['lmd_tf']
        self.tau = config['tau']
        self.sim = config['sim']

        self.tau_plus = config['tau_plus']
        self.beta = config['beta']

        self.fusion_type = config['fusion_type']
        self.lamdas = config['lamdas']

        self.tao_f = config['tao_f']
        self.tao_t = config['tao_t']

        # load dataset info
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.item_mean_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.item_cov_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        self.position_mean_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.position_cov_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)

        self.gma_trm_encoder = GMA_TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.fdf = FDFEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            config=self.config,
        )
        # CLF4Rec
        self.fft_layer = BandedFourierLayer(self.hidden_size, self.hidden_size, 0, 1, length=self.max_seq_length)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Contrastive Learning
        self.contrastive_learning_layer = CLLayer(self.hidden_size, tau=config['cl_temp'])

        # Fusion Attn
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        # Global Graph Learning
        self.item_adjgraph = external_data["adj_graph"].to(self.device)
        self.user_edges = external_data["user_edges"]
        self.item_simgraph = external_data["sim_graph"].to(self.device)
        self.graph_dropout = config["graph_dropout_prob"]

        self.adj_graph_test = external_data["adj_graph_test"].to(self.device)
        self.sim_graph_test = external_data["sim_graph_test"].to(self.device)

        self.gcn = GCN(self.hidden_size, self.hidden_size, self.graph_dropout)

        self.layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.loss_fct = nn.CrossEntropyLoss()

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' be CE!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def item_crop(self, item_seq, item_seq_len, eta=0.6):
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(0, item_seq_len - num_left)
        croped_item_seq = np.zeros(item_seq.shape[0])
        if crop_begin + num_left < item_seq.shape[0]:
            croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(croped_item_seq, dtype=torch.long, device=item_seq.device),\
               torch.tensor(num_left, dtype=torch.long, device=item_seq.device)

    def item_mask(self, item_seq, item_seq_len, gamma=0.3):
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        masked_item_seq[mask_index] = self.n_items  # token 0 has been used for semantic masking
        return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

    def item_reorder(self, item_seq, item_seq_len, beta=0.6):
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = item_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len


    def augment(self, item_seq, item_seq_len):
        aug_seq1 = []
        aug_len1 = []
        for seq, length in zip(item_seq, item_seq_len):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = self.item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = self.item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = self.item_reorder(seq, length)

            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)

        return torch.stack(aug_seq1), torch.stack(aug_len1)

    def _subgraph_agreement(self, aug_g, raw_output_all, raw_output_seq, valid_items_flatten):
        # here it firstly removes items of the sequence in the cooccurrence graph, and then performs the gnn aggregation, and finally calculates the item-wise agreement score.
        aug_output_seq = self.gcn_forward(g=aug_g)[valid_items_flatten]
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(
            self.item_adjgraph, aug_g, valid_items_flatten, raw_output_all)

        view1_sim = F.cosine_similarity(
            raw_output_seq, aug_output_seq, eps=1e-12)
        view2_sim = F.cosine_similarity(
            raw_output_seq, foreign_nbr_ro, eps=1e-12)
        view3_sim = F.cosine_similarity(
            civil_nbr_ro, foreign_nbr_ro, eps=1e-12)
        agreement = (view1_sim+view2_sim+view3_sim)/3
        agreement = torch.sigmoid(agreement)
        agreement = (agreement - agreement.min()) / \
            (agreement.max() - agreement.min())
        agreement = (self.config["weight_mean"] / agreement.mean()) * agreement
        return agreement

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        # 0在后面的mask, 和原版BERT4Rec不同.
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len
        return sequence

    def gcn_forward(self, g=None):
        item_emb = self.item_embedding.weight
        item_emb = self.dropout(item_emb)
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out+item_emb)

    def forward(self, item_seq, item_seq_len, return_all=False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        mean_sequence_emb = self.item_mean_embedding(item_seq) + self.position_mean_embedding(position_ids)
        cov_sequence_emb = self.item_cov_embedding(item_seq) + self.position_cov_embedding(position_ids)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.gma_trm_encoder(input_emb, extended_attention_mask, mean_sequence_emb, cov_sequence_emb, is_f=0, output_all_encoded_layers=True)

        output_t = trm_output[-1]

        trm_output_f = self.fdf(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_f = trm_output_f[-1]


        if return_all:
            return output_t, output_f
        output_t = self.gather_indexes(output_t, item_seq_len - 1)
        output_f = self.gather_indexes(output_f, item_seq_len - 1)
        return output_t, output_f  # [B H]

    def my_fft(self, seq):
        f = torch.fft.rfft(seq, dim=1)
        amp = torch.absolute(f)
        phase = torch.angle(f)
        return amp, phase

    def multi_hot_embed(self, masked_index, max_length):
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(
            0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        if self.config["graphcl_enable"]:
            return self.calculate_loss_graphcl(interaction)

        item_seq = interaction[self.ITEM_SEQ]
        last_item = interaction[self.ITEM_ID]
        masked_item_seq, pos_items = self.reconstruct_train_data(
                item_seq, last_item=last_item)
        seq_output = self.forward(masked_item_seq)

        masked_index = (masked_item_seq == self.mask_token)
        # [mask_num, H]
        seq_output = seq_output[masked_index]
        # [item_num, H]
        # [item_num H]
        test_item_emb = self.item_embedding.weight[:self.n_items]
        # [mask_num, item_num]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            input()
        return loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='cos'):

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def calculate_loss_graphcl(self, interaction):
        user_ids = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.ITEM_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        last_items_indices = torch.tensor([i*self.max_seq_length+j for i, j in enumerate(
            item_seq_len - 1)], dtype=torch.long, device=item_seq.device).view(-1)
        # only the last one
        last_items_flatten = item_seq.view(-1)[last_items_indices]
        valid_items_flatten = last_items_flatten
        valid_items_indices = last_items_indices

        # graph view
        masked_g = self.item_adjgraph
        aug_g = graph_augment(self.item_adjgraph, user_ids, self.user_edges)
        iadj_graph_output_raw = self.gcn_forward(masked_g)
        isim_graph_output_raw = self.gcn_forward(self.item_simgraph)
        iadj_graph_output_seq = iadj_graph_output_raw[valid_items_flatten]
        isim_graph_output_seq = isim_graph_output_raw[valid_items_flatten]

        seq_output_t, seq_output_f = self.forward(item_seq, item_seq_len, return_all=False)
        aug_seq_output_t, aug_seq_output_f = self.forward(item_seq, item_seq_len, return_all=True)
        aug_seq_output_t = aug_seq_output_t.view(-1, self.config["hidden_size"])[valid_items_indices]
        # First-stage CL, providing CL weights
        # CL weights from augmentation
        mainstream_weights = self._subgraph_agreement(aug_g, iadj_graph_output_raw, iadj_graph_output_seq, valid_items_flatten)
        # filtering those len=1, set weight=0.5
        mainstream_weights[item_seq_len == 1] = 0.5

        expected_weights_distribution = torch.normal(self.config["weight_mean"], 0.1, size=mainstream_weights.size()).to(self.device)
        # kl_loss = self.config["kl_weight"] * cal_kl(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        # apply log_softmax for input
        kl_loss = self.config["kl_weight"] * cal_kl_1(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        if torch.isnan(kl_loss):
            logging.info("kl_loss: {}".format(kl_loss))
            logging.info("mainstream_weights: {}".format(
                mainstream_weights.cpu().tolist()))
            logging.info("expected_weights_distribution: {}".format(
                expected_weights_distribution.cpu().tolist()))
            raise ValueError("kl loss is nan")

        personlization_weights = mainstream_weights.max() - mainstream_weights

        # contrastive learning
        if self.config["cl_ablation"] == "full":
            # cl_loss_adj = self.contrastive_learning_layer.grace_loss(
            #     aug_seq_output, iadj_graph_output_seq)
            # cl_loss_a2s = self.contrastive_learning_layer.grace_loss(
            #     iadj_graph_output_seq, isim_graph_output_seq)
            # cl_loss_adj = self.contrastive_learning_layer.vanilla_loss_overall(
            #     aug_seq_output, iadj_graph_output_seq, iadj_graph_output_raw)
            # cl_loss_a2s = self.contrastive_learning_layer.vanilla_loss_overall(
            #     iadj_graph_output_seq, isim_graph_output_seq, isim_graph_output_raw)
            cl_loss_adj = self.contrastive_learning_layer.vanilla_loss(aug_seq_output_t, iadj_graph_output_seq)
            cl_loss_a2s = self.contrastive_learning_layer.vanilla_loss(iadj_graph_output_seq, isim_graph_output_seq)
            # cl_loss = (self.config["graphcl_coefficient"] * (mainstream_weights *cl_loss_adj + personlization_weights * cl_loss_a2s)).mean()
            cl_loss = (self.config["graphcl_coefficient"] * (cl_loss_adj + cl_loss_a2s)).mean()
            if torch.isnan(cl_loss):
                logging.error("cl_loss_adj: {}".format(cl_loss_adj.cpu().tolist()))
                logging.error("cl_loss_a2s: {}".format(cl_loss_a2s.cpu().tolist()))
                logging.error("mainstream_weights: {}".format(mainstream_weights.cpu().tolist()))
                logging.error("personlization_weights: {}".format(personlization_weights.cpu().tolist()))
                logging.error("cl loss is nan")
                raise ValueError("cl loss is nan")
        # Fusion After CL
        if self.config["graph_view_fusion"]:
            # seq_output = self.tao_f * seq_output_f + self.tao_t * seq_output_t
            # 3, N_mask, dim
            mixed_x = torch.stack(
                (seq_output_t, iadj_graph_output_raw[last_items_flatten], isim_graph_output_raw[last_items_flatten]), dim=0)
            weights = (torch.matmul(
                mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
            # 3, N_mask, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            seq_output_t = (mixed_x*score).sum(0)
        # [item_num, H]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output_t, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits+1e-8, pos_items)

        # Time-Frequency ssl
        nce_logits_t_f, nce_labels_t_f = self.info_nce(seq_output_f, seq_output_t, temp=self.tau,
                                                       batch_size=seq_output_t.shape[0],
                                                       sim=self.sim)
        nce_loss_t_f = self.nce_fct(nce_logits_t_f, nce_labels_t_f)

        # Frequency ssl
        f_aug_seq_output_amp, f_aug_seq_output_phase = self.my_fft(seq_output_t)
        f_seq_output_amp, f_seq_output_phase = self.my_fft(seq_output_f)

        # Amp ssl
        nce_logits_amp, nce_labels_amp = self.info_nce(f_aug_seq_output_amp, f_seq_output_amp, temp=self.tau,
                                                       batch_size=seq_output_t.shape[0],
                                                       sim=self.sim)
        nce_loss_amp = self.nce_fct(nce_logits_amp, nce_labels_amp)

        # Phase ssl
        nce_logits_phase, nce_labels_phase = self.info_nce(f_aug_seq_output_phase, f_seq_output_phase, temp=self.tau,
                                                           batch_size=seq_output_t.shape[0],
                                                           sim=self.sim)
        nce_loss_phase = self.nce_fct(nce_logits_phase, nce_labels_phase)
        if torch.isnan(loss):
            logging.error("cl_loss: {}".format(cl_loss))
            logging.error("loss is nan")

        return loss, self.lmd_tf * cl_loss, nce_loss_t_f + nce_loss_amp + nce_loss_phase

    def fast_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        seq_output_t, seq_output_f = self.forward(item_seq, item_seq_len)

        # seq_output = 0.3 * seq_output_f + 0.7 * seq_output_t
        seq_output = self.tao_f * seq_output_f + self.tao_t * seq_output_t

        if self.config["graph_view_fusion"]:
            last_items_flatten = torch.gather(
                item_seq, 1, (item_seq_len - 1).unsqueeze(1)).squeeze()
            # graph view
            masked_g = self.adj_graph_test
            iadj_graph_output_raw = self.gcn_forward(masked_g)
            iadj_graph_output_seq = iadj_graph_output_raw[last_items_flatten]
            isim_graph_output_seq = self.gcn_forward(self.sim_graph_test)[
                last_items_flatten]
            # 3, N_mask, dim
            mixed_x = torch.stack(
                (seq_output, iadj_graph_output_seq, isim_graph_output_seq), dim=0)
            weights = (torch.matmul(
                mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
            # 3, N_mask, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            seq_output = (mixed_x*score).sum(0)

        test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(1), test_item_emb.transpose(1, 2)).squeeze()
        return scores

    def get_fused_emb(self):
        item_emb = self.item_embedding.weight
        adj_graph_emb = self.gcn_forward(self.item_adjgraph)
        sim_graph_emb = self.gcn_forward(self.item_simgraph)
        # 3, N_mask, dim
        mixed_x = torch.stack(
            (item_emb, adj_graph_emb, sim_graph_emb), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        fused_emb = (mixed_x*score).sum(0)
        return fused_emb

class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
