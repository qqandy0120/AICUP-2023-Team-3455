# AI CUP Team 3455

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
### Data Intro
- `all_train_data.jsonl`: Official public train data merged (`public_train_0316.jsonl`+`public_train_0522.jsonl`)
- `all_test_data.jsonl`: Official public test data and private test data merged (`public_test_data.jsonl`+`private_test_data.jsonl`)
## [Subtask 1] Document Retreival
For the files generated from this subtask, we've enclosed them in [data](data), so you can simply check whether
- `data/all_es_train_token_10.txt`
- `data/all_es_test_token_10.txt`
- `data/all_train_doc_select_all.jsonl`
- `data/all_test_doc_select_all.jsonl`
- `data/all/all_{mode}_doc{n}/es{m}.jsonl` <-----see [Subtask 1 Step 3]

all files above exist. If so, please jump to [[Subtask 2] Sentence Retreival](#SR)
### [Step 1] Built up elasticsearch
check [search/main](search/main.ipynb), the ipynb have step by step tutorial. <br>
If you successfully generate:
- `data/all_es_train_token_10.txt`
- `data/all_es_test_token_10.txt`

then you have done this step.

### [Step 2] Create index-based document retrieval result
```bash
python doc_rtv.py
```
If you successfully generate:
- `data/all_train_doc_select_all.jsonl`
- `data/all_test_doc_select_all.jsonl`

then you have done this step.

### [Step 3] Concat index-based document retrieval(method in sample code) with elasticsearch(BM-25) document retrieval.
```bash
python concat_doc_rtv.py
```
After this step, `all_{mode}_doc{n}/es{m}.jsonl` will be created, which means the concatenation of Top $n$ wiki pages via index-based method and Top $m$ wiki pages via elasticsearch (BM-25) method, where $0\leq n, m\leq9$, mode = 'train' or 'test'.

### [Step 4] Check the precision/recall for each combination
```bash
python get_pr.py
```
`pr_results.json` will be created, you can check the precision and recall result there.

## <div id="SR">[Subtask 2] Sentence Retreival</div>
### Input files
- `all_train_data.jsonl`
- `all_test_data.jsonl`
- `all_train_doc9/es9.jsonl` 
- `all_test_doc9/es9.jsonl` 
### Sctipts
- `train_stc_rtv.sh`: Script for Sentence Retreival training
- `validate_stc_rtv.sh`: Script for Sentence Retreival  validation
- `test_stc_rtv.sh`: Script for Sentence Retreival test
### `stc_rtv.py` args
- `exp_name`: experiment name of the setting
- `train_data`: path to all public train data
- `train_doc_data`: Path to public train with predicted pages file from document retrieval 
- `test_data`: path to all test data
- `test_doc_data`: Path to all test with predicted pages file from document retrieval
- `model_name`: Pretrained model name
- `model_ckpt`: checkpoint file name, ***not path**
- `num_epoch`: number of training epoch
- `lr`: learning rate
- `train_batch_size`: batch size for training
- `test_batch_size`: batch size for testing
- `neg_ratio`: possibility to select non-evidence sentence, value = [0,1]
- `validation_step`: how many steps you would like to validate when training
- `seed`: random seed
- `top_n`: how many predicted evidences would in output file
- `do_train`: whether to do training, 1 for yes, 0 for no.
- `do_validate`: whether to do validation, 1 for yes, 0 for no.
- `do_test`: whether to do testing, 1 for yes, 0 for no.
- `do_dynamic_load_neg`: whether to do dynamic load neg dataset, 1 for yes, 0 for no.
- `do_single_evi_train`: whether to concat single evi when training.
- `do_concat_page_name_train`: whether to concat page name in front of evi when training.
- `test_size`: how much ratio to split Dev data from all.
- `freeze_ratio`: how much ratio to freeze the model weight.
- `max_seq_len`: maximum sequence len feed in to the BERT.
- `accumulation_step`: gradient accumulation steps
### [Step 1] Train
To reproduce training checkpoints, run the following code. Or paste it to `train_stc_rtv.sh` and run the script as well. Set on the `do_train` varible to run training.
```bash
bash train_stc_rtv.sh
```
#### ckpt information
- **Target file position [Required to reproduce]** : `checkpoints/sent_retrieval/all_data_len256_lr7_7_top5/model.60000.pt`
- Download: 
We provide two way to reproduce, in case of `gdown` may not work. You should check or create the directory or path on you own if you download ckpt file directly from browser here.

| Ways to download | link or command|
| -------- | -------- |
| Browser | [all_data_len256_lr7_7_top5/](https://drive.google.com/drive/folders/1Uy72mKa9jK6vgIEX7VrBrAxq_K36JA9m?usp=sharing) or  [model.60000.pt](https://drive.google.com/file/d/1M8ziae70YJJ8l7jJKTvCz7EEIELD-vwk/view?usp=sharing) | 
|`gdown`    |`gdown --folder 1Uy72mKa9jK6vgIEX7VrBrAxq_K36JA9m` |

### [Step 2] Validation
Run the following code or paste it to `train_stc_rtv.sh` and run the script as well. We could get training score of train data and validation score of dev data. Also, output files will be use on Subtask 3 training:
- `data/all_data_len256_lr7_7_top5/train_{model_ckpt}_{top_n}.jsonl`
- `data/all_data_len256_lr7_7_top5/dev_{model_ckpt}_{top_n}.jsonl`

```bash
bash validate_stc_rtv.sh
```
### [Step 3] Test
Run the following code or paste it to `test_stc_rtv.sh` and run the script as well. We could get model inference result. We set on the `do_test` varible to run test. Also, output files will be use on Subtask 3 test section:
- `data/all_data_len256_lr7_7_top5/test_{model_ckpt}_{top_n}.jsonl`

```bash
bash test_stc_rtv.sh
```
## [Subtask 3] Claim Verification
### [Step 1]
### [Step 2]
### [Step 3]