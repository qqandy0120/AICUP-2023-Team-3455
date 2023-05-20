import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Set, Tuple, Union
import torch

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}

class BERTDataset(Dataset):
    """BERTDataset class for BERT model.

    Every dataset should subclass it. All subclasses should override `__getitem__`,
    that provides the data and label on a per-sample basis.

    Args:
        data (pd.DataFrame): The data to be used for training, validation, or testing.
        tokenizer (BertTokenizer): The tokenizer to be used for tokenization.
        max_length (int, optional): The max sequence length for input to BERT model. Defaults to 128.
        topk (int, optional): The number of top evidence sentences to be used. Defaults to 5.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        topk: int = 5,
    ):
        """__init__ method for BERTDataset"""
        self.data = data.fillna("")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.topk = topk

    def __len__(self):
        return len(self.data)

class SentRetrievalBERTDataset(BERTDataset):
    """AicupTopkEvidenceBERTDataset class for AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        sentA = item["claim"]
        sentB = item["text"]

        # claim [SEP] text
        concat = self.tokenizer(
            sentA,
            sentB,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}
        if "label" in item:
            concat_ten["labels"] = torch.tensor(item["label"])

        return concat_ten
    
class AicupTopkEvidenceBERTDataset(BERTDataset):
    """AICUP dataset with top-k evidence sentences."""

    def __getitem__(
        self,
        idx: int,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        item = self.data.iloc[idx]
        claim = item["claim"]
        evidence = item["evidence_list"]

        # In case there are less than topk evidence sentences
        pad = ["[PAD]"] * (self.topk - len(evidence))
        evidence += pad
        concat_claim_evidence = " [SEP] ".join([*claim, *evidence])

        concat = self.tokenizer(
            concat_claim_evidence,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        label = LABEL2ID[item["label"]] if "label" in item else -1
        concat_ten = {k: torch.tensor(v) for k, v in concat.items()}

        if "label" in item:
            concat_ten["labels"] = torch.tensor(label)

        return concat_ten