import os
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_VER = {
    'gpt4o': 'gpt-4o-2024-11-20',
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
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--input_idx', type=int, default=-1)
    return parser

def gen_messages(content):
    system_prompt = '''당신은 금융 분야 전문가이다. 답변 이외의 내용은 절대 포함하지 마시오. 답변은 참 또는 거짓 중 하나로 답변하시오.'''
    user_prompt = f'''질문: {content}\n답변:참/거짓'''
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def gen_messages_tmp(content):
    return [
        {"role": "user", "content": content}
    ]

def main(args):
    if args.llm_name in ['gpt4o', 'gpt4o-mini']:
        agent = LLMAgent(LLM_VER[args.llm_name], args.seed, args.max_tokens, args.temperature, args.top_p, args.n, args.frequency_penalty, args.presence_penalty)
    else:
        LLM_NAME = f"{args.llm_name}_{LLM_VER[args.llm_name]}"
        agent = OpenLLMAgent(LLM_NAME, args.seed, args.max_tokens, args.max_new_tokens, args.temperature, args.top_p, args.n, args.frequency_penalty, args.presence_penalty)

    # with open('./data/selected_qa_list.txt', 'r', encoding='utf-8') as f:
    #     input_query = f.read().split('\n')

    with open('./data/batch_files/finfairnessqa_output_guard_task_0.jsonl', 'r', encoding='utf-8') as f:
        input_query = [json.loads(line.strip()) for line in f]
    
    input_query = [i['body']['messages'] for i in input_query]

    response_list = {}
    if args.input_idx == -1:
        for i, q in tqdm(enumerate(input_query), total=len(input_query)):
            # messages = gen_messages(q)
            messages = q
            response = agent.get_response(messages)
            response_list[i] = response
    else:
        q = input_query[args.input_idx]
        messages = gen_messages(q)
        response = agent.get_response(messages)

        response_list[args.input_idx] = response
        print(f"Input Query: {q}")
        print(f"Response: {response}")
    
    # Load existing responses if file exists
    if os.path.exists(f'./results/response_biasQA_output_guardrails_{args.llm_name}.json'):
        with open(f'./results/response_biasQA_output_guardrails_{args.llm_name}.json', 'r', encoding='utf-8') as f:
            existing_responses = json.load(f)
        # Update with new responses
        existing_responses.update(response_list)
        response_list = existing_responses
    
    # Save updated responses
    with open(f'./results/response_biasQA_output_guardrails_{args.llm_name}.json', 'w', encoding='utf-8') as f:
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