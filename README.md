# AI CUP

## [Optional] Create Conda Environment
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
## [Subtask 1]Document Retreival
For the files generated from this subtask, we've enclosed them in [data](data), so you can simple check whether
- data/all_es_train_token_10.txt
- data/all_es_test_token_10.txt
- data/all_train_doc_select_all.jsonl"
- data/all_test_doc_select_all.jsonl"

exist and jump to [[Subtask 2] Sentence Retreival](#L56)
### [Step 1] Built up elasticsearch
check [search/main](search/main.ipynb), the ipynb have step by step tutorial. <br>
If you successfully generate:
- data/all_es_train_token_10.txt
- data/all_es_test_token_10.txt

then you have done this step.

### [Step 2] Create index-based document retrieval result
```bash
python doc_rtv.py
```
If you successfully generate:
- data/all_train_doc_select_all.jsonl"
- data/all_test_doc_select_all.jsonl"

then you have done this step.

### [Step 3] Concat index-based document retrieval(method in sample code) with elasticsearch(BM-25) document retrieval.
```bash
python concat_doc_rtv.py
```
After this step, all_{mode}_doc{n}/es{m}.jsonl will be created, which means the concatenation of top n wiki pages via index-based method and top m wiki pages via elasticsearch(BM-25) method.

### [Step 4] Check the precision/recall for each combination
```bash
python get_pr.py
```
pr_results.json will be created, you can check the result there.

## [Subtask 2] Sentence Retreival
### [Step 1]
### [Step 2]
### [Step 3]
## [Subtask 2] Sentence Retreival
### [Step 1]
### [Step 2]
### [Step 3]