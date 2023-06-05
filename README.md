# AI CUP

## [Optinal] Create Conda Environment
```bash
conda create -n aicup python=3.9
conda activate aicup
```

## Install Package
```bash
pip install -r requirements.in
```

## Download Data
```bash
bash download.sh
```

## Built up elasticsearch
check [search/main](search/main.ipynb), the ipynb have step by step tutorial. <br>
If you successfully produce all_es_train_token_10.txt and all_es_test_token_10.txt, then you have done this step.

## Concat index-based document retrieval with elasticsearch(BM-25) document retrieval.
```bash
python cat_doc_rtv.py
```