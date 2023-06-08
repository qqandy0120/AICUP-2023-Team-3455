import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from collections import Counter
import os
def read_label(jsonl_file):
    jsonl_list = []
    with open(jsonl_file, "r", encoding='utf-8') as f_in:
        # read each line of JSONL file and append it to a list
        tmp = {}
        jsonl_list = []
        for line in f_in:
            tmp = json.loads(line)
            tmp = tmp['predicted_label']
            jsonl_list.append(tmp)
    return jsonl_list

def read_jsonl(jsonl_file):
    jsonl_list = []
    with open(jsonl_file, "r", encoding='utf-8') as f_in:
        # read each line of JSONL file and append it to a list
        jsonl_list = [json.loads(line) for line in f_in]
    return jsonl_list

def main(args):
    jsonl_file_list = list(os.listdir(args.dir))
    pred_list = []
    for f in jsonl_file_list:
        pred_list.append(read_label(f'{args.dir}/{f}'))
    
    transposed_lists = zip(*pred_list)
    ensemble_result = [Counter(column).most_common(1)[0][0] for column in transposed_lists]

    # concat predict sent
    res = read_jsonl(args.sent_file)
    for i in range(len(res)):
        res[i]['predicted_label'] = ensemble_result[i]
    
    with open(args.output_file, "w", encoding='utf-8') as f_out:
        # write the entire list as a single JSON object to output file
        for entry in res:
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write('\n')
    
    # debug
    print(ensemble_result[2])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        required = True
    )
    parser.add_argument(
        "--sent_file",
        type=Path,
        required = True
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default="submission_ensemble.jsonl",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

