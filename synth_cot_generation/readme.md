# Code for CoT generation and Labeling

As we use AWS Bedrock for batch inference, here we provide the code that converts our data to and from Bedrock format. The input data will work with any Llama model on AWS (we use Llama 3.1 8B Instruct for CoT generation and Llama 3.1 70B Instruct for autolabeling).

## CoT Generation

To generate CoT, use `mmlu_cotgen_to_bedrock.py` to create input files for AWS bedrock. Then use `mmlu_cotgen_from_bedrock.py` to parse the Bedrock outputs back into JSON file.

The questions used for training and evaluations are in `mmlu_train_questions.json` and `mmlu_eval_questions.json` respectively.

## Autolabeling

To autolabel the generated CoT, use `mmlu_autolabel_to_bedrock.py` to create input files for AWS bedrock. Then use `mmlu_autolabel_from_bedrock.py` to parse the Bedrock outputs back into JSON file.

## Counterfactual Augementation

For counterfactual augmentation, see the seprate folder `counterfactual_augmentation` in the repo root directory.