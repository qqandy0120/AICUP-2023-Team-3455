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

## Create index-based document retrieval result
```bash
```

## Concat index-based document retrieval(method in sample code) with elasticsearch(BM-25) document retrieval.
```bash
python concat_doc_rtv.py
```
After this step, all_{mode}_doc{n}/es{m}.jsonl will be created, which means the concatenation of top n wiki pages via index-based method and top m wiki pages via elasticsearch(BM-25) method.

## TO check the precision/recall for each combination
```bash
python get_pr.py
```
pr_results.json will be created, you can check the result there.