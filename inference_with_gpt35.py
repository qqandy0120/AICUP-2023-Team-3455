import openai
import json
import os
import random
import tiktoken
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
openai.api_key = "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" # zinc google account
# openai.api_key = "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" # zinc csie account
model_engine = "gpt-3.5-turbo-0301"

def num_tokens_from_messages(prompt, model=model_engine):
    """ Returns the number of tokens used by a list of messages. code modified from openai document """
    user_input = [
            {"role": "user", "content" : prompt}
        ]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in user_input:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens+35

def get_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content" : prompt}
        ]
    )
    return response

def compute_acc(args, candidates, responses):
    # print(f'candidates = {candidates}')
    # print(f'responses = {responses}')
    total_len = len(candidates)
    acc = 0
    f = open(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}/result.txt', 'a')
    for i in range(total_len):
        # parse label and evidence
        cur_labels = []
        cur_evidences = []
        for response in responses[i]:
            # Label
            cur_labels.append(
                'NOT ENOUGH INFO' if '資料不足' in response \
                else ('supports' if '正確' in response \
                else ('refutes' if '錯誤' in response  else 'NOT ENOUGH INFO')) \
            )
            # Evidence
            cur_evidences.extend(re.split(r"[。：；]", response))

        ans_label = 'NOT ENOUGH INFO' # 0 -> not enough info, 1 -> supports, 2 -> refutes
        for label in cur_labels:
            if label == 'supports' or label == 'refutes':
                ans_label = label
                break
        
        # if label is not correct, no need to check evidence
        if ans_label != candidates[i]['label']: 
            f.write(f'*****\n[ans]\nlabel: {cur_labels}\n[Validation] Label wrong!\n')
            continue

        # if is neither, than no need to see the evidence
        if ans_label == 'NOT ENOUGH INFO':
            f.write('[Validation] correct (NOT ENOUGH INFO)\n')
            acc += 1
            continue

        # deal the evidence
        ans_evidence = False
        title_id_pair_list = []
        for evi in cur_evidences:
            if re.search(r'[0-9]+', evi):
                title_id_pair = re.split('，', evi)
                title = title_id_pair[0]
                id_list = []
                if '「' in title and '」' in title:
                    title = title[title.find('「')+1:title.find('」')]
                elif '〈' in title and '〉' in title:
                    title = title[title.find('〈')+1:title.find('〉')]
                elif '《' in title and '》' in title:
                    title = title[title.find('《')+1:title.find('》')]
                title = title.replace('篇', '')
                try:
                    id_list = re.findall(r"[0-9]+", title_id_pair[1])
                    for idx in id_list:
                        title_id_pair_list.append([[title, int(idx)]])
                except:
                    continue

        # format the evidence [strategy could be improved]
        if len(title_id_pair_list) < 5:
            tmp = [[x[0], y[0]] for x in title_id_pair_list for y in title_id_pair_list if x != y]
            title_id_pair_list.extend(tmp)

        # get first five possible ans
        title_id_pair_list = title_id_pair_list[:5]
        f.write(f'[output]:\nlabel: {ans_label}\nevidence: {title_id_pair_list}\n')
        ok = False
        for cc in candidates[i]['evidence']:
            tmp = [sen[2:] for sen in cc]
            for j in title_id_pair_list:
                if j == tmp:
                    print('[INFO] find match acc += 1')
                    ok = True
        if ok:
            f.write('[Validation] correct\n')
            acc += 1
        f.write('[Validation] not matching any evidences!\n')
    return float(acc)/float(total_len)

def onestep(args):
    args.file_prefix = 'onestep'
    # set some constant
    training_set_len = 3969
    token_limit = 4096

    # load preprocessed data
    with open(args.preprocess_data, 'r') as f:
        claim_candidates = json.load(f)
    
    # record prompt prefix
    try:
        os.mkdir(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}')
    except:
        print('the dir exists')
        exit(1)

    # prompt prefix and suffix
    prompt_prefix = ''
    prompt_suffix = '答案：\n\t敘述驗證結果：＿＿＿。\n\t證據：＿＿＿。'
    with open('1step_prefix.txt', 'r') as f:
        prompt_prefix = f.read()
    
    # fix the blank space to save the usage of tokens
    prompt_prefix = prompt_prefix.replace(' ', '')
    
    # choose candidates
    if args.test_one_label:
        # [randomly choose neither or one evi]
        candidates_id_list = []
        for i in range(args.data_len):
            flag = False
            while flag is False:
                tmp = random.randint(0, training_set_len)
                if claim_candidates[tmp]['label'] == 'NOT ENOUGH INFO':
                    candidates_id_list.append(tmp)
                    break
                for evi in claim_candidates[tmp]['evidence']:
                    if len(evi) == 1:
                        flag = True
                if flag is True:
                    candidates_id_list.append(tmp)
    else:
        # [randomly choose n candidates]
        candidates_id_list = random.sample([i for i in range(training_set_len)],args.data_len)
    print(f'select id : {candidates_id_list}')

    candidates_list = []
    responses_list = []
    for i in candidates_id_list:
        # format the candidate
        candidate = claim_candidates[i]
        candidates_list.append(candidate)

        # used for very long data
        prompt_list = ["",]
        prompt_idx = 0
        prompt_list[prompt_idx] = '敘述：' + f'{candidate["claim"]}\n' + '資料：\n'
        default_token = num_tokens_from_messages(prompt_prefix) + num_tokens_from_messages(prompt_list[prompt_idx]) + num_tokens_from_messages(prompt_suffix)
        cur_token = default_token
        for title in candidate['data']:
            tmp = f'\t篇名：{title}\n\t內容：'
            for sen in candidate['data'][title]:
                tmp += f'[{sen}]{candidate["data"][title][sen]}'
            tmp += '\n'
            tmp = tmp.replace('\"', '').replace(':', '').replace(' ', '')
            tmp_token_count = num_tokens_from_messages(tmp)
            if cur_token + tmp_token_count < token_limit:
                prompt_list[prompt_idx] += tmp
                cur_token += tmp_token_count
            else: 
                # prompt too long, create another prompt for this claim
                print('[INFO] create extra prompt for len limit')
                prompt_list[prompt_idx] = prompt_list[prompt_idx].replace('\"', '').replace(':', '').replace(' ', '')
                prompt_list.append("")
                prompt_idx += 1
                cur_token = default_token + tmp_token_count
                prompt_list[prompt_idx] = '敘述：' + f'{candidate["claim"]}\n' + '資料：\n' + tmp
        
        # post processing for the last prompt
        prompt_list[prompt_idx] = prompt_list[prompt_idx].replace('\"', '').replace(':', '').replace(' ', '')

        # write all prompt in txt to see the format
        with open(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}/prompt_{i}.txt', 'a', encoding='utf-8') as f:
            for p in prompt_list:
                f.write(prompt_prefix + f'{p}' + prompt_suffix + '\n\n')

        # concat with the prompt
        chatbot_response_list = []
        output_list = []
        for prompt in prompt_list:
            user_input = prompt_prefix + f'{prompt}' + prompt_suffix
            try:
                chatbot_response = get_response(user_input)
                chatbot_response.update({'label' : candidate['label']})
                chatbot_response.update({'evidence' : candidate['evidence']})
                chatbot_response.update({'prompt' : user_input})
                chatbot_response.update({'prompt_len' : len(user_input)})
            except:
                print('[Token error] This should not happen....')
                chatbot_response = 'token_error'
            chatbot_response_list.append(chatbot_response['choices'][0]['message']['content'])
            output_list.append(chatbot_response)
        responses_list.append(chatbot_response_list)

        # write response
        with open(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}/response{i}.json', 'a') as f:
            json.dump(output_list, f, indent=2, ensure_ascii=False)
        print(f'finished generated response of training data_{i}')

    # compute accuracy
    test_accuracy = compute_acc(args, candidates_list, responses_list)
    print(f'[INFO] result acc = {test_accuracy}')
    with open(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}/prefix.txt', 'w', encoding='utf-8') as f:
        f.write(prompt_prefix)
        f.write(f'\n{test_accuracy}')

def twostep(args):
    args.file_prefix = 'twostep'
    # set some constant
    training_set_len = 3969
    token_limit = 4096

    # load preprocessed data
    with open(args.preprocess_data, 'r') as f:
        claim_candidates = json.load(f)
    training_set_len = len(claim_candidates)
    # record prompt prefix
    try:
        os.mkdir(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}')
    except:
        print('the dir exists')
        exit(1)

    # prompt prefix and suffix
    first_prompt_prefix = ''
    first_prompt_suffix = '答案：\n\t敘述驗證結果：＿＿＿。\n'
    second_prompt_prefix = ''
    second_prompt_suffix = ''
    with open('2step_prefix_first.txt', 'r') as f:
        first_prompt_prefix = f.read()
    with open('2step_prefix_second.txt', 'r') as f:
        second_prompt_prefix = f.read()

    # fix the blank space to save the usage of tokens
    first_prompt_prefix = first_prompt_prefix.replace(' ', '')
    second_prompt_prefix = second_prompt_prefix.replace(' ', '')
    
    # choose candidates
    if args.test_one_label:
        # [randomly choose neither or one evi]
        candidates_id_list = []
        for i in range(args.data_len):
            flag = False
            while flag is False:
                tmp = random.randint(0, training_set_len)
                if claim_candidates[tmp]['label'] == 'NOT ENOUGH INFO':
                    candidates_id_list.append(tmp)
                    break
                for evi in claim_candidates[tmp]['evidence']:
                    if len(evi) == 1:
                        flag = True
                if flag is True:
                    candidates_id_list.append(tmp)
    else:
        # [randomly choose n candidates]
        candidates_id_list = random.sample([i for i in range(training_set_len)],args.data_len)
    print(f'select id : {candidates_id_list}')

    candidates_list = []
    responses_list = []
    for i in candidates_id_list:
        # format the candidate
        candidate = claim_candidates[i]
        candidates_list.append(candidate)

        # used for very long data
        prompt_list = ["",]
        prompt_idx = 0
        prompt_list[prompt_idx] = '敘述：' + f'{candidate["claim"]}\n' + '事實資料：\n'
        default_token = num_tokens_from_messages(first_prompt_prefix) + num_tokens_from_messages(prompt_list[prompt_idx]) + num_tokens_from_messages(first_prompt_suffix)
        cur_token = default_token
        sen_idx = 1
        tmp = ''
        # test BERT sentence selection
        for sent in candidate['predicted_evidence']:
            tmp += f'{sen_idx}. {candidate["data"][sent[0]][str(sent[1])]}\n'
            sen_idx += 1
        tmp += '\n'
        tmp = tmp.replace('\"', '').replace(':', '').replace(' ', '')
        # for title in candidate['data']:
        #     tmp = f'\t篇名：{title}\n\t內容：'
        #     for sen in candidate['data'][title]:
        #         tmp += f'{candidate["data"][title][sen]}'
        #     tmp += '\n'
        #     tmp = tmp.replace('\"', '').replace(':', '').replace(' ', '')
        tmp_token_count = num_tokens_from_messages(tmp)
        if cur_token + tmp_token_count < token_limit:
            prompt_list[prompt_idx] += tmp
            cur_token += tmp_token_count
        else: 
            # prompt too long, create another prompt for this claim
            print('[INFO] create extra prompt for len limit')
            prompt_list[prompt_idx] = prompt_list[prompt_idx].replace('\"', '').replace(':', '').replace(' ', '')
            prompt_list.append("")
            prompt_idx += 1
            cur_token = default_token + tmp_token_count
            prompt_list[prompt_idx] = '敘述：' + f'{candidate["claim"]}\n' + '事實資料：\n' + tmp
        
        # post processing for the last prompt
        prompt_list[prompt_idx] = prompt_list[prompt_idx].replace('\"', '').replace(':', '').replace(' ', '')

        # write all prompt in txt to see the format
        with open(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}/prompt_{i}.txt', 'a', encoding='utf-8') as f:
            for p in prompt_list:
                f.write(first_prompt_prefix + f'{p}' + first_prompt_suffix + '\n\n')

        # concat with the prompt
        chatbot_response_list = []
        output_list = []
        for prompt in prompt_list:
            user_input = first_prompt_prefix + f'{prompt}' + first_prompt_suffix
            try:
                chatbot_response = get_response(user_input)
                chatbot_response.update({'label' : candidate['label']})
                chatbot_response.update({'evidence' : candidate['evidence']})
                chatbot_response.update({'prompt' : user_input})
                chatbot_response.update({'prompt_len' : len(user_input)})
                chatbot_response_list.append(chatbot_response['choices'][0]['message']['content'])
                output_list.append(chatbot_response)
            except:
                print('[Token error] This should not happen....')
                chatbot_response = 'token_error'
        responses_list.append(chatbot_response_list)

        # write response
        with open(f'./cache/{args.file_prefix}/response_v{args.version}_{args.version_name}/response{i}.json', 'a') as f:
            json.dump(output_list, f, indent=2, ensure_ascii=False)
        print(f'finished generated response of training data_{i}')

def chat(args):
    return None


def main(args):
    print(f'[INFO] using model {model_engine}')
    if args.step_option == 0:
        chat(args)
    elif args.step_option == 1:
        onestep(args)
    else:
        twostep(args)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--step_option",
        type=int,
        help="0: chat with gpt3.5, 1: one step method, 2: two step method",
        default=2,
        # required=True
    )
    parser.add_argument(
        "--preprocess_data",
        type=Path,
        help="Path to the train file.",
        default="./cache/train_sentence_concat.json",
        # required=True
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        help="Path to the train file.",
        default="onestep",
    )
    parser.add_argument(
        "--data_len",
        type=int,
        help="The training data len wanted to test.",
        default=10,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--version",
        type=int,
        help="prompt version",
        required=True,
    )
    parser.add_argument(
        "--version_name",
        type=Path,
        help="experiment title",
        required=True,
    )
    parser.add_argument(
        "--test_one_label",
        type=bool,
        help="experiment title",
        required=False,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)