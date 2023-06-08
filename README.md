# AI CUP Team 3455
![PDF](poster.png)
## [Optional] Create Conda Environment
```bash
conda create -n aicup python=3.9
conda activate aicup
```

## Install Package
```bash
pip install -r requirements.in
```

## Download all Data and ckpt
**Here will download a zip file. All file required when reproduce is in it! It for one not reproducing training part.**
There are also seperate download link below for each task. </br>
or download by link: [data/ and checkpoints/](https://drive.google.com/file/d/1XFWUIFrkUogNHNCuWvJaz-5Ec1N32UaK/view?usp=drive_link)
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
all files above exist. If so, please jump to [[Subtask 2] 
Sentence Retreival](#SR)
#### Download data method: 
Browser download: [data/](https://drive.google.com/drive/folders/1PgU4oYxrV5bAliosHQzcvB_EHoNfBr96?usp=sharing)
gdown: `gdown --folder 1PgU4oYxrV5bAliosHQzcvB_EHoNfBr96`
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
For this section, we use our best result as the example of all command.
### Input files
- `all_train_data.jsonl`
- `all_test_data.jsonl`
- `all_train_doc9/es9.jsonl` 
- `all_test_doc9/es9.jsonl` 
### Scripts
- `train_stc_rtv.sh`: Script for Sentence Retreival training
- `validate_stc_rtv.sh`: Script for Sentence Retreival  validation
- `test_stc_rtv.sh`: Script for Sentence Retreival test
### `stc_rtv.py` args (included in Scripts)
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
- `do_t2s`: whether to convert traditional Chinese to simplified Chinese
- `do_concat_page_name_train`: whether to concat page name in front of evi when training.
- `test_size`: how much ratio to split Dev data from all.
- `freeze_ratio`: how much ratio to freeze the model weight.
- `max_seq_len`: maximum sequence length feed in model.
- `accumulation_step`: gradient accumulation steps
### [Step 1] Train
To reproduce training checkpoints, Run default code with the following command.
```bash
bash train_stc_rtv.sh
```
#### Output
Several ckpt files will be dump in `checkpoints/sent_retrieval/{exp_name}/` directory.
#### ckpt information
- **Target ckpt file position [Required to reproduce]** : `checkpoints/sent_retrieval/all_data_len256_lr7_7_top5/model.60000.pt`
#### Ckpt Download: 
We provide two way to reproduce, in case of `gdown` may not work. You should check or create the directory or path on you own if you download ckpt file directly from browser here.

| Ways to download | link or command|
| -------- | -------- |
| Browser download | [all_data_len256_lr7_7_top5/](https://drive.google.com/drive/folders/1Uy72mKa9jK6vgIEX7VrBrAxq_K36JA9m?usp=sharing)   **OR**    [model.60000.pt](https://drive.google.com/file/d/1M8ziae70YJJ8l7jJKTvCz7EEIELD-vwk/view?usp=sharing) | 
|`gdown`    |`gdown --folder 1Uy72mKa9jK6vgIEX7VrBrAxq_K36JA9m` |

### [Step 2] Validate
Run default code with the following command to get the Top k (default set to 10) predicted evidences about train and dev data. We could get training score about train data and validation score about dev data to the model we choose.

```bash
bash validate_stc_rtv.sh
```
#### Output
Output files about train and dev will be use on Subtask 3 training:
- `data/all_data_len256_lr7_7_top5/train_{model_ckpt}_{top_n}.jsonl`
- `data/all_data_len256_lr7_7_top5/dev_{model_ckpt}_{top_n}.jsonl`
### [Step 3] Test
Run default code with the following command. We could get model inference result.

```bash
bash test_stc_rtv.sh
```
#### Output
Output files about test will be use on Subtask 3 testing:
- `data/all_data_len256_lr7_7_top5/test_{model_ckpt}_{top_n}.jsonl`
## [Subtask 3] Claim Verification
For this section, we use our best result as the example of all command. However, the best result is the ensemble of three model from two training.
### Input file (continue example above):
- `train_model.60000.pt_10.jsonl`: 
- `dev_model.60000.pt_10.jsonl`
- `test_model.60000.pt_5.jsonl`
### Scripts
- `train_claim.sh`: Script for Claim Verification training
- `validate_claim.sh`: Script for Claim Verification  validation
- `test_claim.sh`: Script for Claim Verification test
### `claim_vrf.py` args (included in Scripts)
- `train_data`: Path to top k predicted sentences training data from [subtasks 2] output
- `dev_data`: Path to top k predicted sentences development data from [subtasks 2] output
- `test_data`: Path to top k predicted sentences testing data from [subtasks 2] output
- `output_file`: Path to submission file.
- `model_name`: Pretrained model name
- `ckpt_name`: checkpoint file name, ***not path**
- `exp_name`: experiment name of the setting
- `num_epoch`: Training total epoch
- `save_checkpoints_limit`: how many Top N checkpoints you would like to save
- `accumulation_step`: gradient accumulation step
- `lr`: Training learning rate
- `train_batch_size`: batch size for training
- `test_batch_size`: batch size for testing
- `validation_step`: how many steps you would like to validate when training
- `seed`: random seed
- `max_seq_len`: maximum sequence length feed in model,
- `top_n`: use Top N sentence from model for training
- `do_train`: whether to do training, 1 for yes, 0 for no.
- `do_validate`: whether to do validation, 1 for yes, 0 for no.
- `do_test`: whether to do testing, 1 for yes, 0 for no.
- `do_ensemble`: whether to do ensemble, 1 for yes, 0 for no.
- `do_concat_page_name`: whether to concat page name when training **and** testing ensemble, 1 for yes, 0 for no.
- `do_ensemble_topk`: use Top K checkpoints to ensemble.
### [Step 1] Train
To reproduce training checkpoints, run default code with the following command. The default code reproduce for `0.808354_model.58000.pt` and `0.800983_model.64000.pt`
```bash
bash train_claim.sh
```
#### Output
Several ckpt files will be dump in `checkpoints/claim_verification/{exp_name}/` directory.
#### Ckpt information
- **Target ckpt file position [Required to reproduce]** : 
If the following models are loaded, Three models would included in `fix_claim_concat_pgname_ensemble_e125_bs8_3.73e-05_top5_hfl/chinese-macbert-large/` 
    - `0.808354_model.58000.pt`
    - `0.803440_model.69000.pt`
    - `0.800983_model.64000.pt`
#### Ckpt Download: 
We provide two way to reproduce, in case of `gdown` may not work. You should check or create the directory or path on you own if you download ckpt file directly from browser here.
**If download those models sucessfully, you could jump to [Subtask 3 Step 3] do inference to reproduce.**
| Ways to download | link or command|
| -------- | -------- |
| Browser download | [fix_claim_concat_pgname_ensemble_e125_bs8_3.73e-05_top5_hfl/](https://drive.google.com/drive/folders/1Ak-QpzIeMSNAO-jrCcAYRKkSfDpNcUv1?usp=sharing) | 
|`gdown`    |`gdown --folder 1Ak-QpzIeMSNAO-jrCcAYRKkSfDpNcUv1` |

### [Step 2] Validate
Run default code with the following command to get get training score about train data and validation score about dev data to the model ckpt we choose.

```bash
bash validate_claim.sh
```
### [Step 3] Test
Run default code with the following command. We could get model inference result, the final submission.

```bash
bash test_claim.sh
```
#### Output
- `submission.jsonl`
## Other helper program
- `json_convert.py`:
    - turn json to jsonl
    - turn jsonl to json
    - Merge 2 jsonl file
    - truncate jsonl to Top K evidence
- `json_voting.py`: If we have lots of submission, we could use this program to ensemble all models' output label by voting.
## Other checkpoint
## Final Result
**Public Leaderboard score = 0.658241** </br>
**Private Leaderboard score = 0.75041** </br>
**Rank 1** </br>

