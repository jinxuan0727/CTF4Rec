#!/bin/bash

python run_DCRec.py --dataset='reddit' --model='CTF4Rec' --lmd_sem=0.1 --contrast='us_x' --sim='dot' --lmd_tf=0.7  --MAX_ITEM_LIST_LENGTH=60
