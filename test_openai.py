import openai
import json
import os
import random
import tiktoken
from argparse import ArgumentParser, Namespace
from pathlib import Path
openai.api_key = "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" #google account
# openai.api_key = "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" #csie account
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
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def get_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content" : prompt}
        ]
    )
    return response


def main(args):
    print(f'[INFO] using model {model_engine}')
    
    # set some constant
    training_set_len = 3969
    token_limit = 4096

    # load preprocessed data
    with open(args.preprocess_data, 'r') as f:
        claim_candidates = json.load(f)
    
    # record prompt prefix
    try:
        os.mkdir(f'./cache/response_v{args.version}_{args.version_name}')
    except:
        print('the dir exists')
        exit(1)

    # prompt prefix and suffix
    prompt_prefix = ''
    prompt_suffix = '答案：\n\t敘述：＿＿＿。\n\t證據：＿＿＿。'
    with open('1_prefix.txt', 'r') as f:
        prompt_prefix = f.read()
    with open(f'./cache/response_v{args.version}_{args.version_name}/prefix.txt', 'w', encoding='utf-8') as f:
        f.write(prompt_prefix)
    prompt_prefix = prompt_prefix.replace(' ', '')
    
    # choose candidates
    candidates_list = random.sample([i for i in range(training_set_len)],args.data_len)
    for i in candidates_list:
        
        # format the candidate
        candidate = claim_candidates[i]

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
        with open(f'./cache/response_v{args.version}_{args.version_name}/prompt_{i}.txt', 'a', encoding='utf-8') as f:
            for p in prompt_list:
                f.write(prompt_prefix + f'{p}' + prompt_suffix + '\n\n')

        # concat with the prompt
        
        chatbot_response_list = []
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
            chatbot_response_list.append(chatbot_response)
        with open(f'./cache/response_v{args.version}_{args.version_name}/response{i}.json', 'a') as f:
            for chatbot_response in chatbot_response_list:
                json.dump(chatbot_response, f, indent=2, ensure_ascii=False)
        print(f'finished generated response of training data_{i}')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--preprocess_data",
        type=Path,
        help="Path to the train file.",
        default="./cache/train_sentence_concat.json",
        # required=True
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)