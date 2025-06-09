# Bias QA Framework

This repository contains tools and scripts for analyzing and evaluating bias in Large Language Models (LLMs) through question-answering tasks, with a specific focus on financial domain bias.

## Overview

The framework provides functionality to:
	•	Generate and evaluate bias-related questions
	•	Interact with various LLM models (OpenAI GPT models and open-source models)
	•	Process and analyze responses for bias detection
	•	Handle financial domain-specific bias analysis

## Dataset and Processing Pipeline

### Dataset Source
	•	Uses the shchoice/finance-legal-mrc dataset from Hugging Face: https://huggingface.co/datasets/shchoice/finance-legal-mrc

### Data Processing Steps
	1.	Initial Data Processing (load_hf_dataset_n_query_prompt_gen.ipynb)
	•	Loading the shchoice/finance-legal-mrc dataset and preprocessing
	•	Extracting the financial domain subset
	•	Generating initial QA pairs using gpt_api.py for bias-related question creation and filtering
	2.	Manual Curation
	•	Reviewed and refined generated pairs by hand
	•	Constructed a final set of 85 high-quality bias QA examples
	•	[Example QA pairs will be added here]
	3.	Prompt Generation (load_hf_dataset_n_query_prompt_gen.ipynb)
	•	Created input prompts for QA evaluation
	•	Designed and implemented input guardrail prompts to ensure controlled generation
	4.	LLM Response Generation (run_bias_qa_openai_batch.ipynb)
	•	Used GPT-4o via run_bias_qa_openai_batch.ipynb to generate model responses
	5.	Output Guardrail Testing (run_output_guard_openai_batch.ipynb)
	•	Conducted experiments on output guardrails for safe and reliable answers

## Directory Structure

```
bias_QA/
├── data/                               # Raw and processed data files
├── results/                            # Final evaluation outputs and analysis results
├── bias_qa.py                          # Main script for bias QA evaluation
├── gpt_api.py                          # API interface for LLM interactions and QA generation/filtering
├── run_bias_qa_openai_batch.ipynb      # Notebook for batch processing with OpenAI GPT-4o
├── run_output_guard_openai_batch.ipynb # Notebook for output guardrail experiments
└── load_hf_dataset_n_query_prompt_gen.ipynb # Notebook for dataset loading, preprocessing, and prompt generation
```

## Key Components

### 1. LLM Integration
	•	Supports multiple LLMs:
	•	OpenAI (GPT-4, GPT-4o)
	•	Open-source (Llama, Phi-4)
	•	Configurable parameters: temperature, top_p, max_tokens, etc.

### 2. Bias QA Generation
	•	Financial domain bias question creation via gpt_api.py
	•	Automated filtering followed by manual quality control
	•	Final curated set of 85 QA pairs

### 3. Prompt Engineering
	•	Input prompt templates for QA tasks
	•	Guardrail prompts to guide LLM outputs

### 4. Evaluation Pipeline
	•	Batch response generation with GPT-4o
	•	Output guardrail testing to ensure safe, unbiased answers
	•	Results saved under results/

## Usage

### Basic Evaluation Command

```
python bias_qa.py --llm_name gpt4o --temperature 0.0 --n 1
```

### Notebooks
- Dataset & Prompt Prep: load_hf_dataset_n_query_prompt_gen.ipynb
- Response Generation: run_bias_qa_openai_batch.ipynb
- Guardrail Testing: run_output_guard_openai_batch.ipynb

### Results

All final outputs, logs, and analysis summaries are located in the results/ directory.