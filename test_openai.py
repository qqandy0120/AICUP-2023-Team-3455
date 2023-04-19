import openai
import json
openai.api_key = "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" #google account
# openai.api_key = "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" #csie account
model_engine = "gpt-3.5-turbo-0301"

prompt_prefix =' \
    給定一個dictionary，其中包含claim以及數個candidates，claim為一個錯誤或正確的敘述，candidates為數篇我們查到的相關文章，每個candidates由數組{篇名 : {句子id : 句子}} 組成。\n \
    請考慮所有candidates，以中文選擇claim為正確或錯誤，並選出支持的句子作為證據，句子以 "篇名, 句子id" 表示，不需說明理由。\n'


def get_response(prompt):
    # Get the response from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Extract the response from the response object
    return response


def main():
    print(f'[INFO] using model {model_engine}')
    
    # load candidates
    with open('./cache/train_sentence_concat.json', 'r') as f:
        claim_candidates = json.load(f)

    # record prompt prefix
    with open(f'./cache/response_v1/prefix.json', 'w') as f:
        json.dump(prompt_prefix, f, indent=2, ensure_ascii=False)
    
    # choose candidates
    for i in range(3):
        candidate = claim_candidates[i]
        # concat with the prompt
        user_input = prompt_prefix + f'{candidate}'
        
        chatbot_response = get_response(user_input)
        with open(f'./cache/response_v1/response{i}.json', 'w') as f:
            json.dump(chatbot_response, f, indent=2, ensure_ascii=False)
        print(f'finished generated prompt {i}')

main()