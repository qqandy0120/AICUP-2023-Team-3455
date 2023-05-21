import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

def main(args):
    Path('cache/').mkdir(parents=True, exist_ok=True)
    if args.option == 0:
        jsonl_to_json(args)
    else:
        json_to_jsonl(args)

def jsonl_to_json(args):
    with open(args.jsonl_file, "r", encoding='utf-8') as f_in, open(args.json_file, "w", encoding='utf-8') as f_out:
        # read each line of JSONL file and append it to a list
        jsonl_list = [json.loads(line) for line in f_in]
        # write the entire list as a single JSON object to output file
        json.dump(jsonl_list, f_out, indent=2, ensure_ascii=False)

def json_to_jsonl(args):
    with open(args.json_file, "r", encoding='utf-8') as f_in, open(args.jsonl_file, "w", encoding='utf-8') as f_out:
        # write the entire list as a single JSON object to output file
        json_data = json.load(f_in)
        for entry in json_data:
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write('\n')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=Path,
        default='./cache/public_train.json',
    )
    parser.add_argument(
        "--jsonl_file",
        type=Path,
        default="./data/public_train.jsonl",
    )
    parser.add_argument(
        "--option",
        type=int,
        help='0: jsonl to json, 1:json to jsonl',
        default=0
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
