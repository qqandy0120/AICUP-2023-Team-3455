CUDA_VISIBLE_DEVICES="0,1" python3 claim_vrf.py \
--train_data data/all_data_len256_lr7_7_top5/train_model.60000.pt_9.jsonl  \
--test_data data/all_data_len256_lr7_7_top5/test_model.60000.pt_9.jsonl \
--dev_data data/all_data_len256_lr7_7_top5/dev_model.60000.pt_9.jsonl \
--output_file submission_0601_macbert_large_Train_step3_Top9.jsonl \
--model_name hfl/chinese-macbert-large \
--exp_name all_data_best_step2_train_top9 \
--test_batch_size 8 \
--train_batch_size 8 \
--validation_step 1000 \
--seed 3513 \
--num_epoch 125 \
--max_seq_len 512 \
--lr 8.3e-5 \
--accumulation_step 32 \
--do_train 1 \
--do_validate 0 \
--do_test 0 \
--do_ensemble 1 \
--do_ensemble_topk 5 \
--save_checkpoints_limit 25 \