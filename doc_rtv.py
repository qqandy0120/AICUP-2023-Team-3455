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
from tqdm import tqdm

# our own libs
from utils import load_json
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm
import random
from dataset import AicupTopkEvidenceBERTDataset
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)
from argparse import ArgumentParser, Namespace
from collections import Counter
import opencc
import os 

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=4)

wikipedia.set_lang("zh")
print('[INFO] Preloading wiki database...')
wiki_pages = jsonl_dir_to_df("data/wiki-pages")
wiki_mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
del wiki_pages
print('[INFO] Finish preloading wiki database!')

TRAIN_DATA = load_json("data/all_train_data.jsonl")
TEST_DATA = load_json("data/all_test_data.jsonl")
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

@dataclass
class Claim:
    data: str

@dataclass
class AnnotationID:
    id: int

@dataclass
class EvidenceID:
    id: int

@dataclass
class PageTitle:
    title: str

@dataclass
class SentenceID:
    id: int

@dataclass
class Evidence:
    data: List[List[Tuple[AnnotationID, EvidenceID, PageTitle, SentenceID]]]


def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)
    return CONVERTER_S2T.convert(simplified)


def get_nps_hanlp(
    predictor: Pipeline,
    d: Dict[str, Union[int, Claim, Evidence]],
) -> List[str]:
    claim = d["claim"]
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves()))
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
    ]
    return nps


def calculate_precision(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
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


def calculate_recall(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
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


def save_doc(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
    mode: str = "train",
    num_pred_doc: int = 5,
) -> None:
    with open( f"data/all_{mode}_doc_select_all.jsonl", "w", encoding="utf8",) as f:
        for i, d in enumerate(data):
            d["predicted_pages"] = list(predictions.iloc[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def get_pred_pages(
    series_data: pd.Series, 
    ) -> Set[Dict[int, str]]:
    results = []
    tmp_muji = []
    # wiki_page: its index showned in claim
    mapping = {}
    claim = series_data["claim"]
    nps = series_data["hanlp_results"]
    first_wiki_term = []

    for i, np in enumerate(nps):
        # Simplified Traditional Chinese Correction
        wiki_search_results = [
            do_st_corrections(w) for w in wikipedia.search(np)
        ]

        # Remove the wiki page's description in brackets
        wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]
        wiki_df = pd.DataFrame({
            "wiki_set": wiki_set,
            "wiki_results": wiki_search_results
        })

        # Elements in wiki_set --> index
        # Extracting only the first element is one way to avoid extracting
        # too many of the similar wiki pages
        grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
        candidates = grouped_df["wiki_results"].tolist()
        # muji refers to wiki_set
        muji = grouped_df.index.tolist()

        for prefix, term in zip(muji, candidates):
            if prefix not in tmp_muji:
                matched = False

                # Take at least one term from the first noun phrase
                if i == 0:
                    first_wiki_term.append(term)

                # Walrus operator :=
                # https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
                # Through these filters, we are trying to figure out if the term
                # is within the claim
                if (((new_term := term) in claim) or
                    ((new_term := term.replace("·", "")) in claim) or
                    ((new_term := term.split(" ")[0]) in claim) or
                    ((new_term := term.replace("-", " ")) in claim)):
                    matched = True
                elif "·" in term:
                    splitted = term.split("·")
                    for split in splitted:
                        if (new_term := split) in claim:
                            matched = True
                            break

                if matched:
                    # post-processing
                    term = term.replace(" ", "_")
                    term = term.replace("-", "")
                    
                    # eliminate out of wiki db pages
                    if term in wiki_mapping.keys():
                        results.append(term)
                        mapping[term] = claim.find(new_term)
                        tmp_muji.append(new_term)
    # print(mapping)
    ## select all
    results = sorted(mapping, key=mapping.get)
    return results
    # print(results)  
    # instance = {
    #     'id': series_data['id'],
    # }
    # if mode == 'train':
    #     instance.update({'label' : series_data['label']})
    # instance.update({'claim': series_data['claim']})
    # instance.update({"predicted_pages": results})
    # with open( f"data/all_{mode}_doc_select_all.jsonl", "a", encoding="utf8",) as f:
    #     f.write(json.dumps(instance, ensure_ascii=False) + "\n")
    # topk = 10
    # if len(final_results) > topk:
    #     assert -1 not in mapping.values()
    #     results = sorted(mapping, key=mapping.get)[:topk] 
    # elif len(final_results) < 1:
    #     results = first_wiki_term
    # instance['predicted_pages'] = results
    # with open( f"data/all_{mode}_doc_{topk}.jsonl", "a", encoding="utf8",) as f:
    #     f.write(json.dumps(instance, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ## set up HanLP predictor
    predictor = (hanlp.pipeline().append(
    hanlp.load("FINE_ELECTRA_SMALL_ZH"),
    output_key="tok",
    ).append(
        hanlp.load("CTB9_CON_ELECTRA_SMALL"),
        output_key="con",
        input_key="tok",
    ))
    print(f'train data len = {len(TRAIN_DATA)}')
    print(f'test data len = {len(TEST_DATA)}')

    # Step 1. Get noun phrases from hanlp consituency parsing tree
    ## create parsing tree
    hanlp_results = []
    hanlp_file = f"data/hanlp_con_results.pkl"
    if Path(hanlp_file).exists():
        print(f'[INFO] loading from {hanlp_file}')
        with open(hanlp_file, "rb") as f:
            hanlp_results = pickle.load(f)
    else:
        for d in tqdm(TRAIN_DATA, total=len(TRAIN_DATA)):
            hanlp_results.append(get_nps_hanlp(predictor, d))
        print(f'[INFO] creating {hanlp_file}')
        with open(hanlp_file, "wb") as f:
            pickle.dump(hanlp_results, f)

    # hanlp_results = [get_nps_hanlp(predictor, d) for d in TRAIN_DATA]
    # with open(hanlp_file, "wb") as f:
    #     pickle.dump(hanlp_results, f)
        
    train_df = pd.DataFrame(TRAIN_DATA)  # TODO: modify train_data
    train_df.loc[:, "hanlp_results"] = hanlp_results
    predicted_results = train_df.parallel_apply(get_pred_pages, axis = 1)
    save_doc(TRAIN_DATA, predicted_results, mode="train")
    # Step3. Repeat the some processs on test set
    ## create parsing tree
    hanlp_results = []
    hanlp_test_file = f"data/hanlp_con_test_results.pkl"
    if Path(hanlp_test_file).exists():
        print(f'[INFO] loading from {hanlp_test_file}')
        with open(hanlp_test_file, "rb") as f:
            hanlp_results = pickle.load(f)
    else:
        print(f'[INFO] creating {hanlp_test_file}')
        for d in tqdm(TEST_DATA, total=len(TEST_DATA)):
            hanlp_results.append(get_nps_hanlp(predictor, d))
        with open(hanlp_test_file, "wb") as f:
            pickle.dump(hanlp_results, f)
    
    test_df = pd.DataFrame(TEST_DATA)
    test_df.loc[:, "hanlp_results"] = hanlp_results
    test_results = test_df.parallel_apply(get_pred_pages, axis = 1)
    save_doc(TEST_DATA, test_results, mode="test")

    # Step 2. Calculate our results
    # calculate_precision(TRAIN_DATA, predicted_results)
    # calculate_recall(TRAIN_DATA, predicted_results)
