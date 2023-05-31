for i in 5 6 7 8 9 10
do
    python3 json_convert.py --option 2 --topk $i --jsonl_file data/all_data_len256_lr7_7_top5/dev_model.60000.pt_15.jsonl --output_file data/all_data_len256_lr7_7_top5/dev_model.60000.pt_$i.jsonl
    python3 json_convert.py --option 2 --topk $i --jsonl_file data/all_data_len256_lr7_7_top5/test_model.60000.pt_15.jsonl --output_file data/all_data_len256_lr7_7_top5/test_model.60000.pt_$i.jsonl
    python3 json_convert.py --option 2 --topk $i --jsonl_file data/all_data_len256_lr7_7_top5/train_model.60000.pt_15.jsonl --output_file data/all_data_len256_lr7_7_top5/train_model.60000.pt_$i.jsonl
done