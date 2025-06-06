# Gender and Age Bias Experiment

## 1. Introduction
This project is an experiment designed to measure potential gender and age biases present in Large Language Models (LLMs).

## 2. Data
The data used for this experiment can be downloaded from the following link:
[Financial Services AI Competition Data](https://bigdata-finance.kr/dataset/datasetView.do?datastId=SET1400010)

Please download the data and place it in the `gender_age/data` directory.

## 3. Experiment Workflow
The experiment is conducted in the following sequence:

### 3.1. Prompt Generation
First, you need to generate the prompts required for the experiment using the `gender_age/make_prompt.ipynb` notebook. This notebook creates a variety of prompts that include gender and age variables based on the downloaded data.

**Instructions:**
1. Open the `gender_age/make_prompt.ipynb` notebook.
2. Ensure that the data path within the notebook is correctly set to point to the data you downloaded.
3. Run all the cells in the notebook to generate the prompt files. The generated prompts will be saved in the `gender_age/prompt_template` directory.

### 3.2. Running the Experiment with GPT-4o
Once the prompts are generated, you can use the `gender_age/run_batch_api.ipynb` notebook to send them to OpenAI's `gpt-4o` model and retrieve the responses. This notebook utilizes a batch API to process multiple prompts efficiently.

**Instructions:**
1. Open the `gender_age/run_batch_api.ipynb` notebook.
2. Set up your OpenAI API key within the notebook.
3. Specify the path to the prompt file you generated in the previous step.
4. Execute the notebook to run the experiment. The model's responses will be saved to a file in the `gender_age/results` directory.

### 3.3. Performance Evaluation
After the model's responses have been saved, you can evaluate the results of the experiment using the `gender_age/evaluation.py` script. This script calculates various metrics, including accuracy and bias indicators, based on the model's responses.

To run the evaluation, execute the following command in your terminal from the root directory of the project:
```bash
python gender_age/evaluation.py --input_file <path_to_model_response_file>
```
Replace `<path_to_model_response_file>` with the actual path to the file containing the model's responses (e.g., `gender_age/results/output.jsonl`).

### 3.4. Bias Classification with Guardrail Model
After evaluating the performance, you can use the `gender_age/output_guardrail.py` script to classify whether the model's predictions show gender bias. This script utilizes the `kakaocorp/kanana-safeguard-8b` model to analyze pairs of credit score predictions for male and female profiles generated in the previous steps.

To run the bias classification, execute the following command from the project's root directory:
```bash
python gender_age/output_guardrail.py --input_file <path_to_results.csv>
```
