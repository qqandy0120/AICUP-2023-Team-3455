#!/usr/bin/env bash

python3 stc_rtv.py \
--train_data data/all_train_data.jsonl \
--test_data data/all_test_data.jsonl \
--train_doc_data data/all/all_train_doc9/es9.jsonl \
--test_doc_data data/all/all_test_doc9/es9.jsonl \
--exp_name test_new_data \
--model_name hfl/chinese-macbert-large \
--num_epoch 5 \
--train_batch_size 6 \
--test_batch_size 6 \
--validation_step 2000 \
--top_n 5 \
--neg_ratio 0.05 \
--lr 3.7e-5 \
--max_seq_len 512 \
--test_size 0.035 \
--do_t2s 1 \
--do_dynamic_load_neg 1 \
--do_single_evi_train 1 \
--do_concat_page_name_train 1 \
--do_test 0 \
--do_validate 0 \