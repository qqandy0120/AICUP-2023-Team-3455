import typing 
from typing import List, Dict
import copy
import jsonlines
import json
import os
import pandas as pd
from utils import load_json
from doc_rtv import calculate_precision, calculate_recall
from pathlib import Path

def save_list_of_dicts_as_jsonl(data, filename):
    with jsonlines.open(filename, 'w') as writer:
        for item in data:
            writer.write(item)

MODES = ['train', 'test']

for mode in MODES:
    data_path = f'data/all_{mode}_doc_select_all.jsonl'
    with jsonlines.open(data_path, 'r') as f:
        # datas = [line['predicted_pages'] for line in f]
        datas = [line for line in f]

    # with open(data_path, 'r') as f:
    #     datas = json.load(f)

    # datas = [data['predicted_pages'] for data in datas]


    es_path = Path(f'data/all_es_{mode}_token_10.txt')
    with open(es_path, 'r') as f:
        lines = f.readlines()
        es_tokens: List[List[str]] = [line.replace('\n', '').split(' ') for line in lines]


    MIN_DOCN, MAX_DOCN = 0, 9
    MIN_ESN, MAX_ESN = 0, 9

    for doc_n in range(MIN_DOCN, MAX_DOCN+1):
        for es_n in range(MIN_ESN, MAX_ESN+1):
            # print('-------')
            # print(datas[0]['predicted_pages'])
            # print('-------')
            select_tokens = []
            for line in datas:
                select_tokens.append(line['predicted_pages'])
            concat_tokens = []
            for i in range(len(select_tokens)):
                concat_tokens.append(list(set(select_tokens[i][:doc_n]+es_tokens[i][:es_n])))

            out_dir = Path(f'cache/all_{mode}_doc{doc_n}')
            out_dir.mkdir(parents=True, exist_ok=True) 

            result = copy.deepcopy(datas)
            for i, line in enumerate(result):
                line['predicted_pages'] = concat_tokens[i]

            print(os.path.join(out_dir, f'es{es_n}.jsonl'))
            save_list_of_dicts_as_jsonl(result, os.path.join(out_dir, f'es{es_n}.jsonl'))
