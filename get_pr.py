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

def calculate_precision(
    data,
    predictions: pd.Series,
) -> None:
    precision = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])

        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)

        count += 1

    # Macro precision
    print(f"Precision: {precision / count}")

    return precision / count
def calculate_recall(
    data,
    predictions: pd.Series,
) -> None:
    recall = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1

    print(f"Recall: {recall / count}")

    return recall / count

TRAIN_DATA = load_json("data/all_train_data.jsonl")

result = {}
strong_results = {}
MIN_DOCN, MAX_DOCN = 0, 9
MIN_ESN, MAX_ESN = 0, 9

for doc_n in range(MIN_DOCN, MAX_DOCN+1):
    result[f'doc_{doc_n}'] = {}
    for es_n in range(MIN_ESN, MAX_ESN+1):
        with open(f"data/all/all_train_doc{doc_n}/es{es_n}.jsonl", "r", encoding="utf8") as f:
                predicted_results = pd.Series([
                    set(json.loads(line)["predicted_pages"])
                    for line in f
                ])
                p = calculate_precision(TRAIN_DATA, predicted_results)
                r = calculate_recall(TRAIN_DATA, predicted_results)

                result[f'doc_{doc_n}'][f'es_{es_n}'] = {
                    'precision': p,
                    'recall': r
                }

                if p > 0.18 and r > 0.91:
                    strong_results.update({f'doc{doc_n}_es{es_n}': {'precision': p, 'recall': r}})

with open('pr_results.json', 'w') as f:
    json.dump(result, f, indent=2)