import json, os
import random
import argparse
import sys
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import numpy as np


def contains_nan(lst):
    return any(np.isnan(x) for x in lst)


def save_dict_to_file(data, file_path):

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f'successfully saved {file_path}')
    except Exception as e:
        print(f'Error: {e}')


def main():
    
    parser = argparse.ArgumentParser(description='Process rewards for PRM models')

    # Add argument for the directory paths and prm_name
    parser.add_argument('--eval_data_dir', type=str, required=True, help='Directory for example files')
    parser.add_argument('--prm_name', type=str, choices=['Math-Shepherd', 'Math-PSA', 'RLHFlow-Deepseek', 'Qwen-2.5-Math-PRM', 'Llama-PRM800K', 'Qwen-PRM800K', 'VersaPRM', 'VersaPRM-Aug', 'VersaPRM-Math-Subset', 'VersaPRM-Small-Subset', 'VersaPRM-Base-8B', 'VersaPRM-Base-3B'], required=True, help='PRM model to use')
    parser.add_argument('--eval_model_config', type=str, default='./prm_models/model_config.json', help='Path to the evaluation model config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the reward files')
    parser.add_argument('--four_bit', action='store_true')
    parser.add_argument('--sample_4', action='store_true')
    
    # Parse the arguments
    args = parser.parse_args()

    prm_name = args.prm_name
    output_dir = args.output_dir

    # Set the directory paths from the parsed arguments
    eval_data_dir = args.eval_data_dir
    eval_data_file_path_list = [os.path.join(eval_data_dir, file) for file in os.listdir(eval_data_dir)]
    
    # Determine reward file folder based on the `prm_name` argument
    test_dataset_name = eval_data_dir.split('/')[-1]
    reward_file_folder_dir = os.path.join(output_dir, f'{test_dataset_name}_with_{prm_name}_reward')
    os.makedirs(reward_file_folder_dir, exist_ok=True)
    
    # Configure quantization
    if args.four_bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None

    all_file_exists = True

    for eval_data_file_path in eval_data_file_path_list:
        file_save_path = os.path.join(
            reward_file_folder_dir, os.path.basename(eval_data_file_path).split('.js')[0]+f'_with_{prm_name}_rewards.json'
        )

        if not os.path.exists(file_save_path):
            all_file_exists = False

    if not all_file_exists:

        with open(args.eval_model_config, 'r') as file:
            model_config = json.load(file)

        if prm_name == 'Math-Shepherd':
            from prm_models.math_sheperd import MathShepherd
            prm = MathShepherd(
                aggregation='full', 
                quantization_config=quantization_config,
                model_id=model_config[prm_name]['model_id']
            )
        elif prm_name == 'Math-PSA':
            from prm_models.math_psa import MathPSA
            prm = MathPSA(
                aggregation='full', 
                quantization_config=quantization_config,
                model_id=model_config[prm_name]['model_id'],
                downloaded_adapter_path=model_config[prm_name]['downloaded_adapter_path']
            )
        elif prm_name == 'RLHFlow-Deepseek':
            from prm_models.rlhflow_deepseek import RLHflow_Deepseek_8bPRM
            prm = RLHflow_Deepseek_8bPRM(
                aggregation='full', 
                quantization_config=quantization_config,
                model_id=model_config[prm_name]['model_id']
            )
        elif prm_name in ['Qwen-2.5-Math-PRM']:
            from prm_models.qwen25_math_7b_prm800k import QwenMathPRM
            prm = QwenMathPRM(
                aggregation='full',
                model_id=model_config[prm_name]['model_id']
            )
        elif prm_name in ['Qwen-PRM800K']:
            from prm_models.prm_qwen import QwenPRM
            prm = QwenPRM(
                aggregation='full', 
                model_id=model_config[prm_name]['model_id']
            )
        elif prm_name in ['Llama-PRM800K', 'VersaPRM', 'VersaPRM-Aug', 'VersaPRM-Math-Subset', 'VersaPRM-Small-Subset', 'VersaPRM-Base-8B', 'VersaPRM-Base-3B']:
            from prm_models.prm_llama import LlamaPRM
            prm = LlamaPRM(
                aggregation='full', 
                model_id=model_config[prm_name]['model_id']
            ) 
        else:
            raise NotImplementedError

    # Process each example file

    for eval_data_file_path in eval_data_file_path_list:

        file_save_path = os.path.join(
            reward_file_folder_dir, os.path.basename(eval_data_file_path).split('.js')[0]+f'_with_{prm_name}_rewards.json'
        )

        if os.path.exists(file_save_path):
            continue

        with open(eval_data_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Sample 4 examples. For degugging only. 
        if args.sample_4:
            random.seed(42)
            data = random.sample(data, 4)

        for each_data in tqdm(data, desc= f'processing {os.path.basename(eval_data_file_path)}'):

            for cot in each_data['chain_of_thoughts']:

                if 'manually_inspected' in cot.keys():
                    if cot['manually_inspected'] == False:
                        continue

                steps = cot['steps']
                
                if prm_name == 'Math-Shepherd':
                    steps = [step.replace(prm.step_tag, '') for step in steps]
                    ###
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f'\nStep {str(index+1)}: {step} {prm.step_tag}'
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    ###
                    question = each_data['question'].replace(prm.step_tag, '')
                    steps_all = f'{question} ' + ''.join(steps)
                    rewards = prm([steps_all])
                    cot['prm_reward'] = rewards[0].score

                elif prm_name == 'Math-PSA':
                    steps = [step.replace('\n', '') for step in steps]
                    question = each_data['question'].replace('\n', '')

                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f'Step {str(index+1)}: {step} \n\n\n\n\n '
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    steps_all = f'{question} ' + ''.join(steps)
                    rewards = prm([steps_all])
                    cot['prm_reward'] = rewards[0].score

                elif prm_name in ['RLHFlow-Deepseek']:
                    steps = [step.replace('\n', '') for step in steps]
                    question = each_data['question'].replace('\n', '')
                    steps_all = f'{question}\n\n' + '\n\n'.join(steps)
                    rewards = prm([steps_all])
                    cot['prm_reward'] = rewards[0].score
                
                elif prm_name in ['Qwen-2.5-Math-PRM']:
                    steps = cot['steps']
                    steps = [step.replace('<extra_0>', '') for step in steps]
                    question = each_data['question'].replace('<extra_0>', '')
                    messages = [
                        {'role': 'system', 'content': 'Please reason step by step.'},
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': '<extra_0>'.join(steps) + '<extra_0>'},]
                    steps_all = prm.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                    )
                    rewards = prm([steps_all])
                    cot['prm_reward'] = rewards[0].score

                elif prm_name in ['Llama-PRM800K', 'Qwen-PRM800K', 'VersaPRM', 'VersaPRM-Aug', 'VersaPRM-Math-Subset', 'VersaPRM-Small-Subset', 'VersaPRM-Base-8B', 'VersaPRM-Base-3B']:

                    steps = [step.strip().replace(' \n\n\n\n', '') for step in steps]
                    question = each_data['question'].strip().replace(' \n\n\n\n', '')
                    updated_steps = []
                    for index, step in enumerate(steps):
                        indexed_step = f'{step} \n\n\n\n'
                        updated_steps.append(indexed_step)
                    steps = updated_steps
                    steps_all = f'{question} \n\n' + ''.join(steps)
                    rewards = prm([steps_all])
                    cot['prm_reward'] = rewards[0].score

                else:
                    raise NotImplementedError

                if contains_nan(cot['prm_reward']):
                    print(steps_all)
                    print('debugggggggg')
                    rewards = prm([steps_all])
                    raise ValueError

        save_dict_to_file(data=data, file_path=file_save_path)

if __name__ == '__main__':
    main()