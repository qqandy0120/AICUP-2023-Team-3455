# built-in libs
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# 3rd party libs
import hanlp
import opencc
import pandas as pd
import wikipedia
from hanlp.components.pipeline import Pipeline
from pandarallel import pandarallel

# our own libs
from utils import load_json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)

def load_wiki_page():
    pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=6)
    wikipedia.set_lang("zh")
    Path('cache/').mkdir(parents=True, exist_ok=True)
    wiki_pages = jsonl_dir_to_df("data/wiki-pages")
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages)
    del wiki_pages
    return mapping
    #    print(mapping['天王星'])
    ''' (some empty string)
    {'0': '天王星 （ Uranus ） 是一顆在太陽系中離太陽第七近的青色行星 ， 其體積在太陽系中排名第三 、 質量排名第四 。', 
    '1': '', 
    '2': '',
    '3': '天王星的英文名稱Uranus來自古希臘神話的天空之神烏拉諾斯  ， 是克洛諾斯的父親 、 宙斯的祖父 。', 
    '4': '在西方文化中 ， 天王星是太陽系中唯一以希臘神祇命名的行星 ， 其他行星都依照羅馬神祇命名 。', 
    '''

def main(args):
    
    with open(f'{args.cache_dir}/train_doc5sent5.json', 'r') as f:
        training_set = json.load(f)
    
    if args.load_wiki is True:
        print('[INFO] building wiki map')
        mapping = load_wiki_page()
        with open(f'{args.cache_dir}/train_sentence_concat.json', 'w') as f_out:
            train_json_list = []
            for data in training_set:
                claim_and_candidates = {
                    'claim' : data['claim'],
                    'data' : {},
                    'label' : data['label'],
                    'evidence' : data['evidence'],
                    'predicted_evidence' : data['predicted_evidence'],
                    }
                for evi in data['predicted_pages']:
                    try:
                        tmp = mapping[evi]
                    except:
                        continue
                    evi_mapping = {f'{evi}' : {id : mapping[evi][id] for id in mapping[evi] if mapping[evi][id]}}
                    claim_and_candidates['data'].update(evi_mapping)
                train_json_list.append(claim_and_candidates)
            json.dump(train_json_list, f_out, indent=2, ensure_ascii=False)
    
    # show some statistics of data
    cnt_evidence = {0 : 0}
    cnt_evidence_with_label = {'NOT ENOUGH INFO' : {0:0}, 'refutes' : {}, 'supports' : {}}
    min_cnt_evidence_without_label = {0 : 0}

    for data in training_set:
        mn = 1e9
        for evi_list in data["evidence"]:
            mn = min(mn, len(evi_list))
            if data['label'] == 'supports' or data['label'] == 'refutes':
                key = len(evi_list)
                if key not in cnt_evidence.keys():
                    cnt_evidence.update({key : 1})
                else:
                    cnt_evidence[key] += 1
                if key not in cnt_evidence_with_label[data['label']].keys():
                    cnt_evidence_with_label[data['label']].update({key : 1})
                else:
                    cnt_evidence_with_label[data['label']][key] += 1
            else:
                min_cnt_evidence_without_label[0] += 1
                cnt_evidence_with_label[data['label']][0] += 1
                cnt_evidence[0] += 1
        if data['label'] == 'supports' or data['label'] == 'refutes':
            if key not in min_cnt_evidence_without_label.keys():
                min_cnt_evidence_without_label.update({mn : 1})
            else:
                min_cnt_evidence_without_label[mn] += 1
    
    # wrtie data to json
    with open(f'{args.cache_dir}/train_stastics.json', 'w+') as f_out:
        f_out.write(f'total data count = {len(training_set)}\n')
        json.dump(cnt_evidence, f_out, indent=2, ensure_ascii=False)
        f_out.write('\n')
        json.dump(cnt_evidence_with_label, f_out, indent=2, ensure_ascii=False)
        f_out.write('\n')
        json.dump(min_cnt_evidence_without_label, f_out, indent=2, ensure_ascii=False)
        f_out.write('\n')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--load_wiki",
        type=bool,
        help="whether to load wiki mapping",
        default=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)