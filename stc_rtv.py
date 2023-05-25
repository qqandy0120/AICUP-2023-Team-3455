# built-in libs
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
import datetime
# third-party libs
import json
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from dataset import SentRetrievalBERTDataset

# local libs
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)
from argparse import ArgumentParser, Namespace

def evidence_macro_precision(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate precision for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of precision)
        [2]: retrieved (denominator of precision)
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    # Return 0, 0 if label is not enough info since not enough info does not
    # contain any evidence.
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # e[2] is the page title, e[3] is the sentence index
        all_evi = [[e[2], e[3]]
                   for eg in instance["evidence"]
                   for e in eg
                   if e[3] is not None]
        claim = instance["claim"]
        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision /
                this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0


def evidence_macro_recall(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all(
            [len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete
                # groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


def evaluate_retrieval(
    probs: np.ndarray,
    df_evidences: pd.DataFrame,
    ground_truths: pd.DataFrame,
    top_n: int = 5,
    exp_name: str = None,
    cal_scores: bool = True,
    save_name: str = None,
) -> Dict[str, float]:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        Dict[str, float]: F1 score, precision, and recall
    """
    df_evidences["prob"] = probs
    top_rows = (
        df_evidences.groupby("claim").apply(
        lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )

    if cal_scores:
        macro_precision = 0
        macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            macro_prec = evidence_macro_precision(instance, top_rows)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        pr = (macro_precision /
              macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall /
               macro_recall_hits) if macro_recall_hits > 0 else 0.0
        f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        save_dir = os.path.join('data', exp_name)
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        with open(f"{save_dir}/{save_name}", "w", encoding='utf-8') as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[
                    top_rows["claim"] == claim]["predicted_evidence"].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        return {"F1 score": f1, "Precision": pr, "Recall": rec}


def get_predicted_probs(
    model: nn.Module,
    dataloader: Dataset,
    device: torch.device,
) -> np.ndarray:
    """Inference script to get probabilites for the candidate evidence sentences

    Args:
        model: the one from HuggingFace Transformers
        dataloader: devset or testset in torch dataloader

    Returns:
        np.ndarray: probabilites of the candidate evidence sentences
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

    return np.array(probs)


def pair_with_wiki_sentences(
    mapping: Dict[str, Dict[int, str]],
    df: pd.DataFrame,
    negative_ratio: float,
) -> pd.DataFrame:
    """Only for creating train sentences."""
    claims = []
    sentences = []
    labels = []

    # positive
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue

        claim = df["claim"].iloc[i]
        evidence_sets = df["evidence"].iloc[i]
        for evidence_set in evidence_sets:
            sents = []
            for evidence in evidence_set:
                # evidence[2] is the page title
                page = evidence[2].replace(" ", "_")
                # the only page with weird name
                if page == "臺灣海峽危機#第二次臺灣海峽危機（1958）":
                    continue
                # evidence[3] is in form of int however, mapping requires str
                sent_idx = str(evidence[3])
                sents.append(mapping[page][sent_idx])

            whole_evidence = " ".join(sents)

            claims.append(claim)
            sentences.append(whole_evidence)
            labels.append(1)

    # negative
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]

        evidence_set = set([(evidence[2], evidence[3])
                            for evidences in df["evidence"][i]
                            for evidence in evidences])
        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [
                    (page, sent_idx) for sent_idx in mapping[page].keys()
                ]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                if pair in evidence_set:
                    continue
                text = mapping[page][pair[1]]
                # `np.random.rand(1) <= 0.05`: Control not to add too many negative samples
                if text != "" and np.random.rand(1) <= negative_ratio:
                    claims.append(claim)
                    sentences.append(text)
                    labels.append(0)

    return pd.DataFrame({"claim": claims, "text": sentences, "label": labels})


def pair_with_wiki_sentences_eval(
    mapping: Dict[str, Dict[int, str]],
    df: pd.DataFrame,
    is_testset: bool = False,
) -> pd.DataFrame:
    """Only for creating dev and test sentences."""
    claims = []
    sentences = []
    evidence = []
    predicted_evidence = []

    # negative
    for i in range(len(df)):
        # if df["label"].iloc[i] == "NOT ENOUGH INFO":
        #     continue
        claim = df["claim"].iloc[i]

        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for page_name, sentence_id in page_sent_id_pairs:
                text = mapping[page][sentence_id]
                if text != "":
                    claims.append(claim)
                    sentences.append(text)
                    if not is_testset:
                        evidence.append(df["evidence"].iloc[i])
                    predicted_evidence.append([page_name, int(sentence_id)])

    return pd.DataFrame({
        "claim": claims,
        "text": sentences,
        "evidence": evidence if not is_testset else None,
        "predicted_evidence": predicted_evidence,
    })


def main(args):

    pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
    SEED = 42
    TRAIN_DATA = load_json(args.train_data)
    TEST_DATA = load_json(args.test_data)
    DOC_DATA = load_json(args.train_doc_data)
    LABEL2ID: Dict[str, int] = {
        "supports": 0,
        "refutes": 1,
        "NOT ENOUGH INFO": 2,
    }
    ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
    _y = [LABEL2ID[data["label"]] for data in TRAIN_DATA]
    # GT means Ground Truth
    TRAIN_GT, DEV_GT = train_test_split(
        DOC_DATA,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
        stratify=_y,
    )

    # preload wiki database
    print('Preloading wiki database...')
    wiki_pages = jsonl_dir_to_df("data/wiki-pages")
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages)
    del wiki_pages
    print('Finish preloading wiki database!')

    # Step 1. Setup training environment
    #@title  { display-mode: "form" }
    MODEL_NAME = args.model_name  #@param {type:"string"}
    NUM_EPOCHS = args.num_epoch  #@param {type:"integer"}
    LR = args.lr  #@param {type:"number"}
    TRAIN_BATCH_SIZE = args.train_batch_size  #@param {type:"integer"}
    TEST_BATCH_SIZE = args.test_batch_size #@param {type:"integer"}
    NEGATIVE_RATIO = args.neg_ratio  #@param {type:"number"}
    VALIDATION_STEP = args.validation_step  #@param {type:"integer"}
    TOP_N = args.top_n  #@param {type:"integer"}

    EXP_DIR = input("experiment name:")
    if not EXP_DIR:
        EXP_DIR = "sent_retrieval/"+str(datetime.now())
    else:
        EXP_DIR = "sent_retrieval/" + EXP_DIR 
    LOG_DIR = "logs/" + EXP_DIR
    CKPT_DIR = "checkpoints/" + EXP_DIR
    LOG_FILE = CKPT_DIR + "/log.txt"
    ARG_FILE = CKPT_DIR + "/arg.json"
    if not Path(LOG_DIR).exists():
        Path(LOG_DIR).mkdir(parents=True)

    if not Path(CKPT_DIR).exists():
        Path(CKPT_DIR).mkdir(parents=True)


    with open(ARG_FILE, 'w') as f:
        # f.write(str(args))
        json.dump(vars(args), f, indent=2)

    # Step 2. Combine claims and evidences
    train_df = pair_with_wiki_sentences(
        mapping,
        pd.DataFrame(TRAIN_GT),
        NEGATIVE_RATIO,
    )
    counts = train_df["label"].value_counts()
    print("[INFO] Now using the following train data with 0 (Negative) and 1 (Positive)")
    print(counts)

    dev_evidences = pair_with_wiki_sentences_eval(mapping, pd.DataFrame(DEV_GT))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SentRetrievalBERTDataset(train_df, tokenizer=tokenizer)
    val_dataset = SentRetrievalBERTDataset(dev_evidences, tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
         batch_size=TRAIN_BATCH_SIZE,
    )
    eval_dataloader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE)

    del train_df
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'[INFO] using device {device}')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)



    # Step 3. Start training
    if args.do_train == 1:
        print('[INFO] running training section')
        writer = SummaryWriter(LOG_DIR)
        progress_bar = tqdm(range(num_training_steps))
        current_steps = 0

        for epoch in range(NUM_EPOCHS):
            model.train()

            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                writer.add_scalar("training_loss", loss.item(), current_steps)

                y_pred = torch.argmax(outputs.logits, dim=1).tolist()
                y_true = batch["labels"].tolist()

                current_steps += 1

                if current_steps % VALIDATION_STEP == 0 and current_steps > 0:
                    print("[INFO] Training: Start validation")
                    probs = get_predicted_probs(model, eval_dataloader, device)

                    val_results = evaluate_retrieval(
                        probs=probs,
                        df_evidences=dev_evidences,
                        ground_truths=DEV_GT,
                        top_n=TOP_N,
                    )
                    with open(LOG_FILE, "a") as log_out:
                        log_out.write(f'{current_steps}: {val_results}\n')

                    # log each metric separately to TensorBoard
                    for metric_name, metric_value in val_results.items():
                        writer.add_scalar(
                            f"dev_{metric_name}",
                            metric_value,
                            current_steps,
                        )
                    save_checkpoint(model, CKPT_DIR, current_steps)

        print("[INFO] Finished training!")

    if args.do_validate == 1:
        # validation part
        ckpt_name = args.model_ckpt
        print(f'[INFO] loading ckpt from {CKPT_DIR}/{ckpt_name}')
        model = load_model(model, ckpt_name, CKPT_DIR)
        print("[INFO] Start final evaluations and write prediction files.")
        if args.do_train == 1:
            train_evidences = pair_with_wiki_sentences_eval(
                mapping=mapping,
                df=pd.DataFrame(TRAIN_GT),
            )
            train_set = SentRetrievalBERTDataset(train_evidences, tokenizer)
            train_dataloader = DataLoader(train_set, batch_size=TEST_BATCH_SIZE)

            print("[INFO] Start calculating training scores")
            probs = get_predicted_probs(model, train_dataloader, device)
            train_results = evaluate_retrieval(
                probs=probs,
                df_evidences=train_evidences,
                ground_truths=TRAIN_GT,
                top_n=TOP_N,
                exp_name=EXP_DIR.split("/")[1],
                save_name=f"train_{ckpt_name}.jsonl",
            )
            print(f"[INFO] Training scores => {train_results}")

            print("[INFO] Start validation")
            probs = get_predicted_probs(model, eval_dataloader, device)
            val_results = evaluate_retrieval(
                probs=probs,
                df_evidences=dev_evidences,
                ground_truths=DEV_GT,
                top_n=TOP_N,
                exp_name=EXP_DIR.split("/")[1],
                save_name=f"dev_{ckpt_name}.jsonl",
            )
            print(f"[INFO] Validation scores => {val_results}")


        # Step 4. Check on our test data
        test_data = load_json(args.test_doc_data)

        test_evidences = pair_with_wiki_sentences_eval(
            mapping,
            pd.DataFrame(test_data),
            is_testset=True,
        )
        test_set = SentRetrievalBERTDataset(test_evidences, tokenizer)
        test_dataloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE)

        print("[INFO] Start predicting the test data")
        probs = get_predicted_probs(model, test_dataloader, device)
        evaluate_retrieval(
            probs=probs,
            df_evidences=test_evidences,
            ground_truths=test_data,
            top_n=TOP_N,
            cal_scores=False,
            exp_name=EXP_DIR.split("/")[1],
            save_name=f"test_{ckpt_name}.jsonl",
        )

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        help="path to pulic train data",
        default="data/public_train.jsonl",
    )
    parser.add_argument(
        "--train_doc_data",
        type=str,
        help="path to pulic train data",
        default="data/train_doc5.jsonl",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        help="path to public test data",
        default="data/public_test.jsonl"
    )
    parser.add_argument(
        "--test_doc_data",
        type=str,
        help = 'path to doc retrieve test data',
        default='data/test_doc5.jsonl'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="pretrained model name",
        default="bert-base-chinese"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="model.2000.pt"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        help="number epoch",
        default=1
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=2e-5
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="training batch size",
        default=64
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="testing batch size",
        default=256
    )
    parser.add_argument(
        "--neg_ratio",
        type=float,
        help="negative ratio",
        default=0.03
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        help="validation step",
        default=200
    )
    parser.add_argument(
        "--top_n",
        type=int,
        help="choose top n evi",
        default=5
    )
    parser.add_argument(
        "--do_validate",
        type=int,
        help="whether to do validation part",
        default=1
    )
    parser.add_argument(
        "--do_train",
        type = int,
        help="whether to do training part",
        default=1
    )
    args = parser.parse_args()
    return args

# model parameter
# MODEL_NAME = "bert-base-chinese"  #@param {type:"string"}
# NUM_EPOCHS = 1  #@param {type:"integer"}
# LR = 2e-5  #@param {type:"number"}
# TRAIN_BATCH_SIZE = 64  #@param {type:"integer"}
# TEST_BATCH_SIZE = 256  #@param {type:"integer"}
# NEGATIVE_RATIO = 0.03  #@param {type:"number"}
# VALIDATION_STEP = 50  #@param {type:"integer"}
# TOP_N = 5  #@param {type:"integer"}

if __name__ == '__main__':
    args = parse_args()
    main(args)
