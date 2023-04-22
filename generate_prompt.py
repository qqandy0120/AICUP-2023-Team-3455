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

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=6)
wikipedia.set_lang("zh")

from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)

Path('cache/').mkdir(parents=True, exist_ok=True)
wiki_pages = jsonl_dir_to_df("data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages)
del wiki_pages
# print(mapping['天王星'])
''' (some empty string)
{'0': '天王星 （ Uranus ） 是一顆在太陽系中離太陽第七近的青色行星 ， 其體積在太陽系中排名第三 、 質量排名第四 。', 
 '1': '', 
 '2': '',
 '3': '天王星的英文名稱Uranus來自古希臘神話的天空之神烏拉諾斯  ， 是克洛諾斯的父親 、 宙斯的祖父 。', 
 '4': '在西方文化中 ， 天王星是太陽系中唯一以希臘神祇命名的行星 ， 其他行星都依照羅馬神祇命名 。', 
'''

with open('./cache/train_doc5.json', 'r') as f:
    training_set = json.load(f)

with open('./cache/train_sentence_concat.json', 'w') as f_out:
    train_json_list = []
    for data in training_set:
        claim_and_candidates = {
            'claim' : data['claim'],
            'data' : {},
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

