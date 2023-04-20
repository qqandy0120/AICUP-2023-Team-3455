import openai
import json
import os
openai.api_key = "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" #google account
# openai.api_key = "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" #csie account
model_engine = "gpt-3.5-turbo-0301"

with open('prefix.txt', 'r') as f:
    prompt_prefix = f.read()

def get_response(prompt):
    # Get the response from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content" : prompt}
        ]
    )
    # Extract the response from the response object
    return response


def main():
    print(f'[INFO] using model {model_engine}')
    
    # load candidates
    with open('./cache/train_sentence_concat.json', 'r') as f:
        claim_candidates = json.load(f)

    version = 5
    # record prompt prefix
    try:
        os.mkdir(f'./cache/response_v{version}')
    except:
        print('the dir exists')
        exit(1)

    with open(f'./cache/response_v{version}/prefix.txt', 'w', encoding='utf-8') as f:
        f.write(prompt_prefix)
    
    # choose candidates
    id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in id_list:
        candidate = claim_candidates[i]
        # concat with the prompt
        user_input = prompt_prefix + f'{candidate}'
        user_input.replace("candidates", "候選證據文章").replace("claim", "敘述") #not working 

        chatbot_response = get_response(user_input)
        chatbot_response.update({'prompt' : user_input})
        with open(f'./cache/response_v{version}/response{i}.json', 'w') as f:
            json.dump(chatbot_response, f, indent=2, ensure_ascii=False)
        print(f'finished generated response of training data_{i}')

if __name__ == '__main__':
    main()