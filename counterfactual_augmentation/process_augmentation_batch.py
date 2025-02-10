"""
This script processes counterfactual augmentation batches created through create_augmentation_batch.py and ran by Llama 3.1 70B in AWS Bedrock.
Augmented steps are extracted by the script and then combined with the labeled source data to make the final json format.
To use this script, please change the following parameters below.
"""

import json
import re
from tqdm import tqdm

# Change the following 6 parameters 
input_file = 'batch_1.jsonl.out'
out_file = 'mmlu_batch_processed.json'
use_math_error_types = False
base_data_file = 'mmlu_labeled.json'
save_failure_records = False
fail_records_save_file = 'failure_records.json'


valid_error_types = ['conflicting steps', 'non-sequitur', 'factual', 'false assumption', 'contextual', 'calculation', 'incorrect equation']

# Load the augmentation batch output data
data = []
with open(input_file, 'r', encoding="utf-8") as f:
    for i, line in tqdm(enumerate(f)):
        obj = json.loads(line)
        data.append(obj)

# Load the labeled source data used to generate the augmentation batch input
base_data = json.load(open(base_data_file, 'r'))
data_dict = {}
for data_sample in base_data:
    data_dict[data_sample['id']] = data_sample
    for cot in data_dict[data_sample['id']]['chain_of_thoughts']:
        cot['augs'] = []


def find_cot_with_id(cots, id):
    for cot in cots:
        if cot['cot_id'] == id:
            return cot
    return None


# get id/generation pairs for successful and failed 
failed = []
succeeded = []

# the main batch processing loop
for i, sample in tqdm(enumerate(data)):
    id, cot_id = sample['recordId'].split('+')
    
    if (sample.get('modelOutput', -1) == -1):
        continue
    llm_response = sample['modelOutput']['generation']
    source_data = data_dict[id]
    cot = find_cot_with_id(source_data['chain_of_thoughts'], cot_id)['steps']
    question = source_data['question']

    ex = re.search(r'INCORRECT_STEP:\n.*\n+ERROR_EXPLANATION:', llm_response, re.DOTALL)
    if (ex is None):
        failed.append({'id': id, 'fail_pt': 1, 'generated': llm_response})
        continue
    
    incorrect_step = re.search(r'{.*}',ex.group(), re.DOTALL)

    if (incorrect_step is None):
        failed.append({'id': id, 'fail_pt': 2, 'generated': llm_response})
        continue
    incorrect_step = incorrect_step.group()[1:-1]

    step_label = re.search(r'STEP_NUM:.*\n\d+', llm_response)
    if (step_label is None):
        failed.append({'id': id, 'fail_pt': 3, 'generated': llm_response})
        continue
    error_idx = int(re.findall(r'\d+', step_label.group())[0]) - 1

    if (error_idx >= len(cot)):
       failed.append({'id': id, 'fail_pt': 4, 'error_idx': error_idx, 'cot': cot,  'generated': llm_response})
       continue

    error_type = re.search(r'ERROR_TYPE:.*\n.*\n', llm_response)
    if (error_type is None):
      failed.append({'id': id, 'fail_pt': 5, 'error_idx': error_idx, 'generated': llm_response})
      continue

    error_type = error_type.group().replace("ERROR_TYPE:", "").strip().lower()
    error_type = error_type.replace('assumption', 'false assumption').replace('numerical', 'calculation').replace('algebraic', 'calculation').replace('arithmetic', 'calculation').replace('conflicting step', 'conflicting steps').replace('unit', 'incorrect equation').replace('conversion', 'incorrect equation').replace('formula', 'incorrect equation').replace('false definition', 'factual').replace('overgeneralization', 'false assumption').replace('conceptual', 'factual').replace('rounding', 'calculation').replace('methodological', 'factual').replace('computational', 'calculation').replace('contradict', 'conflicting steps').replace('sign error', 'calculation').replace('counting', 'calculation').replace('false classification', 'factual').replace('red herring', 'non-sequitur').replace('false analogy', 'factual')

    error_type_label = ""
    valid_error = False
    for etype in valid_error_types:
       if etype in error_type:
          error_type_label = etype
          valid_error = True
    
    
    if (valid_error == False):
       failed.append({'id': id, 'generated': llm_response, 'error_idx': error_idx, 'fail_pt': 4, 'error_type': error_type})

    succeeded.append({'id': id, 'question': question, 'cot': cot, 'error_idx': error_idx, 'incorrect_step': incorrect_step, 'full_response': llm_response})
    
    find_cot_with_id(source_data['chain_of_thoughts'], cot_id)['augs'].append({'aug_step': incorrect_step,
                                                                               'aug_idx': error_idx,
                                                                               'aug_type': error_type_label 
    })

print(f"Num failed: {len(failed)}")
if (save_failure_records):
   json.dump(failed, open(fail_records_save_file, 'w'), indent=2)

print(f"Num succeeded: {len(succeeded)}")
json.dump(list(data_dict.values()), open(out_file, 'w'), indent=2)