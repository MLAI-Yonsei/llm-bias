import os
import re
import json
import random
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_VER = {
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt41': 'gpt-4.1-2025-04-14',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'llama': "meta-llama/Llama-3.2-3B-Instruct",
    'phi4': 'microsoft/Phi-4-mini-instruct'
}

class OpenLLMAgent():
    def __init__(self, llm_name, seed=0, max_tokens=2048, max_new_tokens=50, temperature=1.0, top_p=1.0, n=1, frequency_penalty=0.0, presence_penalty=0.0, batch_size=256):

        self.llm = llm_name.split('_')[0]
        self.version = llm_name.split('_')[1]

        self.model = AutoModelForCausalLM.from_pretrained(self.version, torch_dtype=torch.bfloat16, cache_dir='./models', device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.version, padding_side="left", cache_dir='./models', truncation_side="left")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_tokens = min(self.tokenizer.model_max_length, max_tokens)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.n = n

        print(f"LLM Generation Setting: max_tokens={self.max_tokens}, max_new_tokens={self.max_new_tokens}")

    def get_response(self, messages: list):
        responses = []
        if self.llm in ['qwen3', 'qwen306b']:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)            
        else:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens).to(self.model.device)

        for _ in tqdm(range(self.n), desc=f"Generating {self.n} responses"):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            batch_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            batch_responses = [r.split('assistant')[-1].strip().replace(' ', '') for r in batch_responses]
            responses.extend(batch_responses)
        return responses

class LLMAgent():
    def __init__(self, llm_name, seed=0, max_tokens=2048, temperature=1.0, top_p=1.0, n=1, frequency_penalty=0.0, presence_penalty=0.0):
        self.llm_name = llm_name
    
        api_key = os.environ.get('OPENAI_API_KEY')
        self.client = OpenAI(
            api_key=api_key,
        )

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self.n = n
        self.seed = seed
    
    def get_response(self, messages: list):
        response = self.client.chat.completions.create(
            seed=self.seed,
            model=self.llm_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=self.n,
        )

        if self.n > 1:
            all_responses = [response.choices[i].message.content for i in range(len(response.choices))]
            return all_responses
        
        return [response.choices[0].message.content]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default='gpt4o')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prompt_type', type=str, default='bias_qa')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--input_idx', type=int, default=-1)
    return parser

def gen_messages_fin_docs(docs_json):
    user_prompt = f"""
다음은 문서 인덱스를 key로, 문서 제목을 value로 가지는 JSON 형식의 데이터입니다:

{docs_json}

이 문서들은 금융 또는 법률 분야에서 수집된 것입니다.
이 중 **금융(finance)** 분야에 해당하며, **대한민국(국내)** 관련 내용만 포함하는 문서만 선별해서 동일한 형식(JSON dictionary)으로 반환해주세요.
오직 금융 분야이며 국내 관련 문서만 포함되어야 하며, 문서의 제목을 기반으로 판단하세요.
결과는 유효한 JSON 형식으로만 출력하세요.
"""
    return [
        {"role": "user", "content": user_prompt}
    ]

def gen_messages_bias_qa(cont):

    user_prompt = f"""다음 질문들 중에서 평가 대상자의 특정 집단에 대한 편향성을 확인하기에 적절한 질문들만 선별해서 동일한 형식(JSON dictionary)으로 반환해주세요.
결과는 유효한 JSON 형식으로만 출력하세요.

{cont}
"""
    
    return [
        {"role": "user", "content": user_prompt}
    ]


def main(args):
    if args.llm_name in ['gpt4o', 'gpt4o-mini', 'gpt41']:
        agent = LLMAgent(LLM_VER[args.llm_name], args.seed, args.max_tokens, args.temperature, args.top_p, args.n, args.frequency_penalty, args.presence_penalty)
    else:
        LLM_NAME = f"{args.llm_name}_{LLM_VER[args.llm_name]}"
        agent = OpenLLMAgent(LLM_NAME, args.seed, args.max_tokens, args.max_new_tokens, args.temperature, args.top_p, args.n, args.frequency_penalty, args.presence_penalty)

    input_query = []
    # Load Data

    if args.prompt_type == 'fin_docs':
        df = pd.read_csv('./data/doc_title.csv')
        
        # Split dataframe into chunks of 100 rows
        chunk_size = 100
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            docs_dict = {str(idx): title for idx, title in zip(chunk.index, chunk['doc_title'])}
            docs_json = json.dumps(docs_dict, ensure_ascii=False, indent=2)
            input_query.append(docs_json)
            
        gen_messages = gen_messages_fin_docs

    elif args.prompt_type == 'bias_qa':
        with open('./data/selected_qa_dict.json', 'r', encoding='utf-8') as f:
            selected_qa_dict = json.load(f)
            
        # Convert dictionary to list of items
        items = list(selected_qa_dict.items())
        
        # Split into chunks of 100
        chunk_size = 100
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            chunk_dict = dict(chunk)
            chunk_json = json.dumps(chunk_dict, ensure_ascii=False, indent=2)
            input_query.append(chunk_json)
        gen_messages = gen_messages_bias_qa
    
    # Gen. response
    response_list = {}
    if args.input_idx == -1:
        for i, q in tqdm(enumerate(input_query), total=len(input_query)):
            messages = gen_messages(q)
            response = agent.get_response(messages)
            response_list[i] = response[0]
    else:
        q = input_query[args.input_idx]
        messages = gen_messages(q)
        response = agent.get_response(messages)

        json_str = re.sub(r"^```json|```$", "", response[0].strip(), flags=re.MULTILINE)
        json_response = json.loads(json_str)

        response_list.update(json_response)
        print(f"Input Query: {q}")
        print(f"Response: {response}")
    
    # Load existing responses if file exists
    if os.path.exists(f'./data/biasQA_filtered_qa.json'):
        with open(f'./data/biasQA_filtered_qa.json', 'r', encoding='utf-8') as f:
            existing_responses = json.load(f)
        # Update with new responses
        existing_responses.update(response_list)
        response_list = existing_responses

    # Save updated responses
    with open(f'./data/biasQA_filtered_qa.json', 'w', encoding='utf-8') as f:
        json.dump(response_list, f, ensure_ascii=False, indent=4)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministc = True
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    main(args)