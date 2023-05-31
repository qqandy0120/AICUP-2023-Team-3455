import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm
import random
import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    get_linear_schedule_with_warmup,
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
from argparse import ArgumentParser, Namespace
from collections import Counter
import opencc
import os 
pandarallel.initialize(progress_bar=True, verbose=0, nb_workers=10)
converter = opencc.OpenCC('t2s.json')  # 't2s.json' for Traditional to Simplified conversion

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def shuffle_evidence(evidence):
    random.shuffle(evidence)
    return evidence

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
        df["evidence"] = df["evidence"].apply(
            lambda x: [[x]] if not isinstance(x[0], list) else [x]
            if not isinstance(x[0][0], list) else x)
    
    # convert claim to simplified chinese
    df['claim'] = df['claim'].apply(lambda x : converter.convert(x))
    print(df['claim'])

    print(f"Extracting evidence_list for the {mode} mode ...")
    if mode == "eval":
        # extract evidence
        df["evidence_list"] = df["predicted_evidence"].apply(lambda x: [
            mapping.get(evi_id, {}).get(str(evi_idx), "")
            for evi_id, evi_idx in x  # for each evidence list
        ][:topk] if isinstance(x, list) else [])
        df['evidence_list'] = df["evidence_list"].apply(lambda x:
            [converter.convert(sent) for sent in x]
        )
        print(df["evidence_list"][:topk])
        print( f'\nrunning {mode} mode\n')
    else:
        # extract evidence [DEFAULT CODE]
        # df["evidence_list"] = df["evidence"].parallel_map(lambda x: [
        #     " ".join([  # join evidence
        #         mapping.get(evi_id, {}).get(str(evi_idx), "")
        #         for _, _, evi_id, evi_idx in evi_list
        #     ]) if isinstance(evi_list, list) else ""
        #     for evi_list in x  # for each evidence list
        # ][:1] if isinstance(x, list) else [])
        df["evidence_list"] = df["evidence"].apply(lambda x: [ [evi_id, evi_idx] 
            for evi_list in x if isinstance(evi_list, list) 
            for _, _, evi_id, evi_idx in evi_list if evi_id != None
        ][:topk] if isinstance(x, list) else [])

        # get ready for concat predicted evi + shuffle
        # TODO: remove the same sent in predicted and evi
        df['concat_predicted_evidence'] = df.apply( lambda row:[
                x for x in row["predicted_evidence"] if x not in row['evidence']
            ],axis = 1
        )
        df['concat_predicted_evidence'] = df['concat_predicted_evidence'].apply( lambda x : random.sample(x, min(len(x), topk)))
        df['concat_predicted_evidence'] = df['concat_predicted_evidence'].apply(shuffle_evidence)

        # concat evi list to [:topk] + shuffle
        df["evidence_list"] = df.apply(lambda row: (row['evidence_list'] + row['concat_predicted_evidence'])[:topk], axis=1)
        df["evidence_list"] = df["evidence_list"].apply(shuffle_evidence)
        df["evidence_list"] = df["evidence_list"].apply(lambda x: [
            mapping.get(evi_id, {}).get(str(evi_idx), "")
            for evi_id, evi_idx in x  # for each evidence list
        ] if isinstance(x, list) else [])
        
        # convert to simplified chinese 
        df['evidence_list'] = df["evidence_list"].apply(lambda x:
            [converter.convert(sent) for sent in x]
        )
        # for debug
        # df['row_length'] = df["evidence_list"].apply(lambda row: len(row)) 
        # print(df['row_length']) 
        print(df["evidence_list"])
        print(f'\nrunning {mode} mode\n')
    return df

def main(args):
    LABEL2ID: Dict[str, int] = {
        "supports": 0,
        "refutes": 1,
        "NOT ENOUGH INFO": 2,
    }
    # print([INFO] current dir f'{os.listdir(".")}')
    ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
    TRAIN_DATA = load_json(args.train_data)
    DEV_DATA = load_json(args.dev_data)
    # TRAIN_PKL_FILE = Path(args.train_pkl_file)
    # DEV_PKL_FILE = Path(args.dev_pkl_file)
    #@title  { display-mode: "form" }
    MODEL_NAME = args.model_name  #@param {type:"string"}
    TRAIN_BATCH_SIZE = args.train_batch_size  #@param {type:"integer"}
    TEST_BATCH_SIZE = args.test_batch_size  #@param {type:"integer"}
    SEED = args.seed  #@param {type:"integer"}
    LR = args.lr  #@param {type:"number"}
    NUM_EPOCHS = args.num_epoch  #@param {type:"integer"}
    MAX_SEQ_LEN = args.max_seq_len  #@param {type:"integer"}
    EVIDENCE_TOPK = args.top_n  #@param {type:"integer"}
    VALIDATION_STEP = args.validation_step  #@param {type:"integer"}
    OUTPUT_FILENAME = args.output_file
    EXP_DIR = f"claim_verification/{args.exp_name}_e{NUM_EPOCHS}_bs{TRAIN_BATCH_SIZE}_" + f"{LR}_top{EVIDENCE_TOPK}" + f'_{MODEL_NAME}'
    LOG_DIR = "logs/" + EXP_DIR
    CKPT_DIR = "checkpoints/" + EXP_DIR
    LOG_FILE = CKPT_DIR + "/log.txt"
    best_score = 0
    BEST_CKPT = args.ckpt_name
    validation_scores = []
    # set random seed
    same_seeds(SEED)

    print('[INFO] Preloading wiki database...')
    wiki_pages = jsonl_dir_to_df("data/wiki-pages")
    mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
    del wiki_pages
    print('[INFO] Finish preloading wiki database!')

    if not Path(LOG_DIR).exists():
        Path(LOG_DIR).mkdir(parents=True)

    if not Path(CKPT_DIR).exists():
        Path(CKPT_DIR).mkdir(parents=True)

    # Step 2. Concat claim and evidences
    train_df = join_with_topk_evidence(
            pd.DataFrame(TRAIN_DATA),
            mapping,
            topk=EVIDENCE_TOPK,
        )

    dev_df = join_with_topk_evidence(
            pd.DataFrame(DEV_DATA),
            mapping,
            mode="eval",
            topk=EVIDENCE_TOPK,
        )

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'[INFO] using device {device}')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    # lr_scheduler = set_lr_scheduler(optimizer, num_training_steps)
    # scheduler with warmup
    warmup_steps = (int)(num_training_steps/100)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    if args.do_train == 1:
        SETTING_FILE = CKPT_DIR + "/setting.sh"
        SH_FILE = 'train_stc_rtv.sh'
        ## save the train exp setting
        print(f'[INFO] saving training exp setting in {SETTING_FILE}')
        with open(SH_FILE, 'rb') as f_in, open(SETTING_FILE, 'wb') as f_out:
            f_out.write(f_in.read())
        writer = SummaryWriter(LOG_DIR)
        progress_bar = tqdm(range(num_training_steps))
        current_steps = 0
        
        # gradient accumulation
        accumulation_steps = args.accumulation_step
        total_loss = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                # gradient accumulation
                loss = loss / accumulation_steps
                total_loss += loss
                loss.backward()

                if (current_steps + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    progress_bar.update(accumulation_steps)
                    writer.add_scalar("training_loss", total_loss, current_steps)
                    total_loss = 0

                y_pred = torch.argmax(outputs.logits, dim=1).tolist()
                y_true = batch["labels"].tolist()
                current_steps += 1

                if current_steps % VALIDATION_STEP == 0 and current_steps > 0:
                    print("[INFO] Start validation")
                    val_results = run_evaluation(model, eval_dataloader, device)
                    # log each metric separately to TensorBoard
                    for metric_name, metric_value in val_results.items():
                        print(f"{metric_name}: {metric_value}")
                        writer.add_scalar(f"{metric_name}", metric_value, current_steps)
                    if best_score < val_results['val_acc']:
                        best_score = val_results['val_acc']
                        #val_acc=0.5453_model.13000.pt \
                        BEST_CKPT = f"{val_results['val_acc']:.4f}_model.{current_steps}.pt"
                    validation_scores.append(f"{val_results['val_acc']:.4f}")
                    save_checkpoint(
                        model,
                        CKPT_DIR,
                        current_steps,
                        mark=f"{val_results['val_acc']:.4f}",
                    )
                    # save validate result to log
                    with open(LOG_FILE, "a") as log_out:
                        log_out.write(f'{current_steps} steps: {val_results}\n')
                    print(f'{current_steps} steps: {val_results}\n')
                    # save top K checkpoints
                    validation_scores = sorted(validation_scores, reverse=True)[:args.save_checkpoints_limit]
                    for file_name in os.listdir(CKPT_DIR):
                        if file_name.endswith('.pt') and file_name.split('_')[0] not in validation_scores:
                            os.remove(os.path.join(CKPT_DIR, file_name))

            #reload dataset
            print(f'[INFO] Reloading train dataset')
            train_df = join_with_topk_evidence(
                pd.DataFrame(TRAIN_DATA),
                mapping,
                topk=EVIDENCE_TOPK,
            )
            train_dataset = AicupTopkEvidenceBERTDataset(
                train_df,
                tokenizer=tokenizer,
                max_length=MAX_SEQ_LEN,
            )
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=TRAIN_BATCH_SIZE,
            )
        print("[INFO] Finished training!")

    # Step 4. Make your submission
    if args.do_validate == 1:
        TEST_DATA = load_json(args.test_data)
        # TEST_PKL_FILE = Path(args.test_pkl_file)
        print(f'[INFO] load test_df from {TEST_DATA}')
        test_df = join_with_topk_evidence(
            pd.DataFrame(TEST_DATA),
            mapping,
            mode="eval",
            topk=EVIDENCE_TOPK,
        )
        test_dataset = AicupTopkEvidenceBERTDataset(
            test_df,
            tokenizer=tokenizer,
            max_length=MAX_SEQ_LEN,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
        if args.do_ensemble == 0:
            # ckpt_name = "val_acc=0.4259_model.750.pt"  #@param {type:"string"}
            model = load_model(model, BEST_CKPT, CKPT_DIR)
            predicted_label = run_predict(model, test_dataloader, device)

            predict_dataset = test_df.copy()
            predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
            predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
                OUTPUT_FILENAME,
                orient="records",
                lines=True,
                force_ascii=False,
            )
        elif args.do_ensemble == 1:
            predict_label_list = []
            predict_dataset = test_df.copy()
            ckpt_list = list(os.listdir(CKPT_DIR))
            ckpt_list.remove('setting.sh')
            ckpt_list.remove('log.txt')
            ckpt_list = sorted(ckpt_list, reverse=True)[:args.do_ensemble_topk]
            for ckpt in ckpt_list:
                print(f'loading {ckpt} to predict\n')
                model = load_model(model, ckpt, CKPT_DIR)
                predicted_label = run_predict(model, test_dataloader, device)
                predict_label_list.append(predicted_label)
                print(predicted_label[0])
            # Transpose the lists using zip()
            transposed_lists = zip(*predict_label_list)
            # Find the most frequent elements at each position
            ensemble_result = [Counter(column).most_common(1)[0][0] for column in transposed_lists]
            predict_dataset["predicted_label"] = list(map(ID2LABEL.get, ensemble_result))
            predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
                OUTPUT_FILENAME,
                orient="records",
                lines=True,
                force_ascii=False,
            )

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # TRAIN_DATA = load_json(args.train_sent_data)
    parser.add_argument(
        "--train_data",
        type=Path,
        help="path to train set data",
        default="data/train_doc5sent5.jsonl",
    )
    # DEV_DATA = load_json("data/dev_doc5sent5.jsonl")
    parser.add_argument(
        "--dev_data",
        type=Path,
        help="path to dev jsonl data",
        default="data/dev_doc5sent5.jsonl",
    )
    # TRAIN_PKL_FILE = Path("data/train_doc5sent5.pkl")
    parser.add_argument(
        "--train_pkl_file",
        type=Path,
        help="path to train pkl data",
        default='None',
    )
    # DEV_PKL_FILE = Path("data/dev_doc5sent5.pkl")
    parser.add_argument(
        "--dev_pkl_file",
        type=Path,
        help="path to dev pkl data",
        default='None'
    )
    # TEST_DATA = load_json("data/test_doc5sent5.jsonl")
    parser.add_argument(
        "--test_data",
        type=Path,
        help="path to evi retrieve data",
        default="data/test_doc5sent5.jsonl"
    )
    # TEST_PKL_FILE = Path("data/test_doc5sent5.pkl")
    parser.add_argument(
        "--test_pkl_file",
        type=Path,
        help = 'path to evi retrieve test pkl file',
        default='123'
    )
    # OUTPUT_FILENAME = "submission.jsonl"
    parser.add_argument(
        "--output_file",
        type=Path,
        help="path to submission file",
        default="submission.jsonl"
    )
    # MODEL_NAME = args.model_name  #@param {type:"string"}
    parser.add_argument(
        "--model_name",
        type=str,
        help="pretrained model name",
        default="bert-base-chinese"
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        help="ckpt name",
        default=""
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="ckpt name",
        required=True
    )
    # NUM_EPOCHS = 20  #@param {type:"integer"}
    parser.add_argument(
        "--num_epoch",
        type=int,
        help="number epoch",
        default=20
    )
    parser.add_argument(
        "--save_checkpoints_limit",
        type=int,
        help="save_checkpoints_limit",
        default=10
    )
    parser.add_argument(
        "--accumulation_step",
        type=int,
        help="gradient accumulation step",
        default=8
    )
    # LR = 7e-5  #@param {type:"number"}
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=7e-5
    )
    # TRAIN_BATCH_SIZE = args.train_batch_size  #@param {type:"integer"}
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="training batch size",
        default=32
    )
    # TEST_BATCH_SIZE = args.test_batch_size  #@param {type:"integer"}
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
    # VALIDATION_STEP = 25  #@param {type:"integer"}
    parser.add_argument(
        "--validation_step",
        type=int,
        help="validation step",
        default=25
    )
    # SEED = 42  #@param {type:"integer"}
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=42
    )
    # MAX_SEQ_LEN = 256  #@param {type:"integer"}
    parser.add_argument(
        "--max_seq_len",
        type=int,
        help="max_seq_len",
        default=512
    )
    # EVIDENCE_TOPK = 5  #@param {type:"integer"}
    parser.add_argument(
        "--top_n",
        type=int,
        help="choose top n evi",
        default=5
    )
    parser.add_argument(
        "-do_validate",
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
    parser.add_argument(
        "--do_ensemble",
        type = int,
        help="whether to do ensemble",
        default=0
    )
    parser.add_argument(
        "--do_ensemble_topk",
        type = int,
        help="how many model to do ensemble",
        default=3
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
