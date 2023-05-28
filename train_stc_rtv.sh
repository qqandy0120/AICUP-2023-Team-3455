#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="3" python3 stc_rtv.py \
--train_doc_data aicup_doc_cache/train_select_all_es/select_all_es_9_avg_12_99.jsonl \
--test_doc aicup_doc_cache/test_select_all_es/test_select_all_es_9_avg_13_22.jsonl \
--exp_name test_loader_t2s \
--model_name hfl/chinese-macbert-large \
--num_epoch 5 \
--train_batch_size 2 \
--test_batch_size 2 \
--validation_step 100 \
--top_n 5 \
--neg_ratio 0.01 \
--lr 6.5e-5 \

