import openai
import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
openai.api_key = "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" #google account
# openai.api_key = "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" #csie account
model_engine = "gpt-3.5-turbo-0301"

with open('prefix.txt', 'r') as f:
    prompt_prefix = f.read()

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
    
    # load preprocessed data
    with open(args.preprocess_data, 'r') as f:
        claim_candidates = json.load(f)
    
    # record prompt prefix
    try:
        os.mkdir(f'./cache/response_v{args.version}_{args.version_name}')
    except:
        print('the dir exists')
        exit(1)

    with open(f'./cache/response_v{args.version}_{args.version_name}/prefix.txt', 'w', encoding='utf-8') as f:
        f.write(prompt_prefix)
    
    suffix = '答案：\n\t敘述正確或錯誤: ＿＿。\n\t證據: ＿＿。」'
    # choose candidates
    id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in id_list:
        # format the candidate
        candidate = claim_candidates[i]
        prompt = '敘述：' + f'{candidate["claim"]}\n' + '資料：\n'
        for title in candidate['data']:
            prompt += f'\t篇名：{title}\n\t內容：'
            for sen in candidate['data'][title]:
                prompt += f'[{sen}]{candidate["data"][title][sen]}'
            prompt += '\n'
        prompt.replace('\"', '').replace(':', '')

        # concat with the prompt
        user_input = prompt_prefix + f'{prompt}'
        chatbot_response = ''
        try:
            chatbot_response = get_response(user_input)
            chatbot_response.update({'prompt' : user_input})
            chatbot_response.update({'prompt_len' : len(user_input)})
        except:
            chatbot_response = 'token_error'
        with open(f'./cache/response_v{args.version}_{args.version_name}/response{i}.json', 'w') as f:
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
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--version",
        type=Path,
        help="experiment version count",
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