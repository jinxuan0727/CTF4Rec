import pandas as pd
import numpy as np

ori_filename = "yelp.inter"
train_file_name = "yelp.train.inter"
test_file_name = "yelp.test.inter"
rating_ = 4

raw_data = pd.read_csv(ori_filename, sep="\t")
print(raw_data)
# print interaction num
print("inter: ", len(raw_data), "U: ", len(raw_data["user_id:token"].unique()), "I: ",
      len(raw_data["item_id:token"].unique()))

raw_data = raw_data.loc[raw_data["rating:float"] > rating_]
print("inter: ", len(raw_data), "U: ", len(raw_data["user_id:token"].unique()), "I: ",
      len(raw_data["item_id:token"].unique()))
# 5-Core Filter
from collections import Counter

user_id_list = raw_data["item_id:token"].tolist()
user_id_count = Counter(user_id_list)
user_id_remove = [item_id for item_id, count in user_id_count.items() if count < 5]
raw_data = raw_data[~raw_data["item_id:token"].isin(user_id_remove)]

user_id_list = raw_data["user_id:token"].tolist()
user_id_count = Counter(user_id_list)
user_id_remove = [user_id for user_id, count in user_id_count.items() if count < 5]
raw_data = raw_data[~raw_data["user_id:token"].isin(user_id_remove)]

print("inter: ", len(raw_data), "U: ", len(raw_data["user_id:token"].unique()), "I: ",
      len(raw_data["item_id:token"].unique()))

raw_data_grouped = raw_data.groupby("user_id:token").agg(
    item_seq=pd.NamedAgg(column='item_id:token', aggfunc=list),
    time_seq=pd.NamedAgg(column='timestamp:float', aggfunc=list)
)

for row in raw_data_grouped.index:
    seq = raw_data_grouped.loc[row, "item_seq"]
    time_seq = np.asarray(raw_data_grouped.loc[row, "time_seq"])
    unique_seq = []
    unique_indicies = []
    for idx, i in enumerate(seq):
        if i not in unique_seq:
            unique_seq.append(i)
            unique_indicies.append(idx)
    time_seq = time_seq[unique_indicies]
    # sort
    sorted_indicies = np.argsort(time_seq)
    raw_data_grouped.at[row, "item_seq"] = np.asarray(unique_seq)[sorted_indicies]
    raw_data_grouped.at[row, "time_seq"] = time_seq[sorted_indicies]

indices_to_drop = list()
for idx in raw_data_grouped.index:
    item_seq = raw_data_grouped.loc[idx, "item_seq"]
    if len(item_seq) < 3 or len(item_seq) > 51:
        indices_to_drop.append(idx)

raw_data_grouped.drop(index=indices_to_drop, inplace=True)
raw_data_grouped.reset_index(inplace=True)

all_items = list()
for idx in raw_data_grouped.index:
    all_items.extend(raw_data_grouped.loc[idx, "item_seq"].tolist())
all_items = list(set(all_items))
item_ids = list(range(1, len(all_items) + 1))
item_remap = dict()
for i in range(len(all_items)):
    item_remap[all_items[i]] = item_ids[i]

raw_data_grouped["item_seq"] = raw_data_grouped["item_seq"].apply(
    lambda x: list(map(lambda y: item_remap[y], x.tolist())))

train_file = open(train_file_name, "w")
test_file = open(test_file_name, "w")

train_file.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
test_file.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")

for idx in raw_data_grouped.index:
    uid = idx
    item_seq = raw_data_grouped.loc[idx, "item_seq"]
    item_seq = list(map(lambda x: str(x), item_seq))
    while len(item_seq) > 51:
        right_seq = item_seq[51:]
        new_seq = item_seq[:51]
        test_file.write(f"{uid}\t" + " ".join(new_seq[:-1]) + f"\t{new_seq[-1]}\n")
        train_file.write(f"{uid}\t" + " ".join(new_seq[:-2]) + f"\t{new_seq[-2]}\n")
        item_seq = right_seq
    if not len(item_seq) < 3:
        test_file.write(f"{uid}\t" + " ".join(item_seq[:-1]) + f"\t{item_seq[-1]}\n")
        train_file.write(f"{uid}\t" + " ".join(item_seq[:-2]) + f"\t{item_seq[-2]}\n")

train_file.close()
test_file.close()
