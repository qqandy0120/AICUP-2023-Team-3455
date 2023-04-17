import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from dataset import AicupTopkEvidenceBERTDataset
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)

pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=4)

def run_evaluation(model: torch.nn.Module, dataloader: DataLoader, device):
    model.eval()

    loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_true.extend(batch["labels"].tolist())

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss += outputs.loss.item()
            logits = outputs.logits
            y_pred.extend(torch.argmax(logits, dim=1).tolist())

    acc = accuracy_score(y_true, y_pred)

    return {"val_loss": loss / len(dataloader), "val_acc": acc}

def run_predict(model: torch.nn.Module, test_dl: DataLoader, device) -> list:
    model.eval()

    preds = []
    for batch in tqdm(test_dl,
                      total=len(test_dl),
                      leave=False,
                      desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(**batch).logits
        pred = torch.argmax(pred, dim=1)
        preds.extend(pred.tolist())
    return preds

def join_with_topk_evidence(
    df: pd.DataFrame,
    mapping: dict,
    mode: str = "train",
    topk: int = 5,
) -> pd.DataFrame:
    """join_with_topk_evidence join the dataset with topk evidence.

    Note:
        After extraction, the dataset will be like this:
               id     label         claim                           evidence            evidence_list
        0    4604  supports       高行健...     [[[3393, 3552, 高行健, 0], [...  [高行健 （ ）江西赣州出...
        ..    ...       ...            ...                                ...                     ...
        945  2095  supports       美國總...  [[[1879, 2032, 吉米·卡特, 16], [...  [卸任后 ， 卡特積極參與...
        停各种战争及人質危機的斡旋工作 ， 反对美国小布什政府攻打伊拉克...

        [946 rows x 5 columns]

    Args:
        df (pd.DataFrame): The dataset with evidence.
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        topk (int, optional): The topk evidence. Defaults to 5.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame: The dataset with topk evidence_list.
            The `evidence_list` column will be: List[str]
    """

    # format evidence column to List[List[Tuple[str, str, str, str]]]
    if "evidence" in df.columns:
        df["evidence"] = df["evidence"].parallel_map(
            lambda x: [[x]] if not isinstance(x[0], list) else [x]
            if not isinstance(x[0][0], list) else x)

    print(f"Extracting evidence_list for the {mode} mode ...")
    if mode == "eval":
        # extract evidence
        df["evidence_list"] = df["predicted_evidence"].parallel_map(lambda x: [
            mapping.get(evi_id, {}).get(str(evi_idx), "")
            for evi_id, evi_idx in x  # for each evidence list
        ][:topk] if isinstance(x, list) else [])
        print(df["evidence_list"][:5])
    else:
        # extract evidence
        df["evidence_list"] = df["evidence"].parallel_map(lambda x: [
            " ".join([  # join evidence
                mapping.get(evi_id, {}).get(str(evi_idx), "")
                for _, _, evi_id, evi_idx in evi_list
            ]) if isinstance(evi_list, list) else ""
            for evi_list in x  # for each evidence list
        ][:1] if isinstance(x, list) else [])

    return df


LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

TRAIN_DATA = load_json("data/train_doc5sent5.jsonl")
DEV_DATA = load_json("data/dev_doc5sent5.jsonl")

TRAIN_PKL_FILE = Path("data/train_doc5sent5.pkl")
DEV_PKL_FILE = Path("data/dev_doc5sent5.pkl")


#@title  { display-mode: "form" }

MODEL_NAME = "bert-base-chinese"  #@param {type:"string"}
TRAIN_BATCH_SIZE = 32  #@param {type:"integer"}
TEST_BATCH_SIZE = 32  #@param {type:"integer"}
SEED = 42  #@param {type:"integer"}
LR = 7e-5  #@param {type:"number"}
NUM_EPOCHS = 20  #@param {type:"integer"}
MAX_SEQ_LEN = 256  #@param {type:"integer"}
EVIDENCE_TOPK = 5  #@param {type:"integer"}
VALIDATION_STEP = 25  #@param {type:"integer"}

OUTPUT_FILENAME = "submission.jsonl"

EXP_DIR = f"claim_verification/e{NUM_EPOCHS}_bs{TRAIN_BATCH_SIZE}_" + f"{LR}_top{EVIDENCE_TOPK}"
LOG_DIR = "logs/" + EXP_DIR
CKPT_DIR = "checkpoints/" + EXP_DIR
TRAIN_PKL_FILE = Path("data/train_doc5sent5.pkl")
DEV_PKL_FILE = Path("data/dev_doc5sent5.pkl")


if __name__ == '__main__':
    print('Preloading wiki database...')
    wiki_pages = jsonl_dir_to_df("data/wiki-pages")
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
    del wiki_pages
    print('Finish preloading wiki database!')

    if not Path(LOG_DIR).exists():
        Path(LOG_DIR).mkdir(parents=True)

    if not Path(CKPT_DIR).exists():
        Path(CKPT_DIR).mkdir(parents=True)


    # Step 2. Concat claim and evidences
    if not TRAIN_PKL_FILE.exists():
        train_df = join_with_topk_evidence(
            pd.DataFrame(TRAIN_DATA),
            mapping,
            topk=EVIDENCE_TOPK,
        )
        train_df.to_pickle(TRAIN_PKL_FILE, protocol=4)
    else:
        with open(TRAIN_PKL_FILE, "rb") as f:
            train_df = pickle.load(f)

    if not DEV_PKL_FILE.exists():
        dev_df = join_with_topk_evidence(
            pd.DataFrame(DEV_DATA),
            mapping,
            mode="eval",
            topk=EVIDENCE_TOPK,
        )
        dev_df.to_pickle(DEV_PKL_FILE, protocol=4)
    else:
        with open(DEV_PKL_FILE, "rb") as f:
            dev_df = pickle.load(f)

    # Step 3. Training
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = AicupTopkEvidenceBERTDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
    )
    val_dataset = AicupTopkEvidenceBERTDataset(
        dev_df,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=TRAIN_BATCH_SIZE,
    )
    eval_dataloader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)

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
                print("Start validation")
                val_results = run_evaluation(model, eval_dataloader, device)

                # log each metric separately to TensorBoard
                for metric_name, metric_value in val_results.items():
                    print(f"{metric_name}: {metric_value}")
                    writer.add_scalar(f"{metric_name}", metric_value, current_steps)

                save_checkpoint(
                    model,
                    CKPT_DIR,
                    current_steps,
                    mark=f"val_acc={val_results['val_acc']:.4f}",
                )

    print("Finished training!")

    # Step 4. Make your submission

    TEST_DATA = load_json("data/test_doc5sent5.jsonl")
    TEST_PKL_FILE = Path("data/test_doc5sent5.pkl")

    if not TEST_PKL_FILE.exists():
        test_df = join_with_topk_evidence(
            pd.DataFrame(TEST_DATA),
            mapping,
            mode="eval",
            topk=EVIDENCE_TOPK,
        )
        test_df.to_pickle(TEST_PKL_FILE, protocol=4)
    else:
        with open(TEST_PKL_FILE, "rb") as f:
            test_df = pickle.load(f)

    test_dataset = AicupTopkEvidenceBERTDataset(
        test_df,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    ckpt_name = "val_acc=0.4259_model.750.pt"  #@param {type:"string"}
    model = load_model(model, ckpt_name, CKPT_DIR)
    predicted_label = run_predict(model, test_dataloader, device)

    predict_dataset = test_df.copy()
    predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
    predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
        OUTPUT_FILENAME,
        orient="records",
        lines=True,
        force_ascii=False,
    )






