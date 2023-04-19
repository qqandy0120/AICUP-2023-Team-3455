import json

jsonl_file = "./data/train_doc5.jsonl"
json_file = "./cache/train_doc5.json"

with open(jsonl_file, "r", encoding='utf-8') as f_in, open(json_file, "w", encoding='utf-8') as f_out:
    # read each line of JSONL file and append it to a list
    jsonl_list = [json.loads(line) for line in f_in]
    # write the entire list as a single JSON object to output file
    json.dump(jsonl_list, f_out, indent=2, ensure_ascii=False)
