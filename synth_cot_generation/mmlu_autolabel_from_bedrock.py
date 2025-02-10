import json
import os
import re
import argparse

def get_data(data_dir):
    '''
    aggregates all jsonl file in directory into one list
    '''
    data = []
    for filename in os.listdir(data_dir):

        if filename.endswith('.jsonl.out'):

            file_path = f'{data_dir}/{filename}'
            with open(file_path, 'r') as f:
                for l in f:
                    data.append(json.loads(l))

    return data

def get_id_to_idx(data):
    id_to_idx = {}

    for q_idx, d in enumerate(data):
        q_id = d['id']
        for cot_idx, cot in enumerate(d['chain_of_thoughts']):
            cot_id = cot['cot_id']

            id_to_idx[f'{q_id}+{cot_id}'] = (q_idx, cot_idx)

    return id_to_idx


def parse_label(model_out):

    pattern = r"boxed\{(-?\d+)\}"
    match = re.search(pattern, model_out)
    if match:
        # Extract and return the number as an integer
        return int(match.group(1))
    return None


if __name__=='__main__':




    parser = argparse.ArgumentParser(description='Script to process bedrock MMLU CoT autolabeling outputs.')
    parser.add_argument('-p','--path', type=str, help='Path to mmlu questions with cot', default='./data/mmlu_train_questions_cot.json')
    parser.add_argument('-b','--bedrockpath', type=str, help='path to directory of bedrock outputs we want to proccess. Shoud contain jsonl.out files', default='./bedrock_outputs/mmlu_train_autolabel')
    parser.add_argument('-o','--outputdir', type=str, help='directory to store proccessed outputs', default='./data')


    args = parser.parse_args()

    with open(args.path, 'r') as f:
        cot_data = json.load(f)




    bedrock_autolabel_data = get_data(args.bedrockpath)

    id_to_idx = get_id_to_idx(cot_data)





    for d in bedrock_autolabel_data:

        id = d['recordId']
        q_id, cot_id = id_to_idx[id]

        cot_len = len(cot_data[q_id]['chain_of_thoughts'][cot_id]['steps'])

        if 'modelOutput' not in d:
            continue

        label = parse_label(d['modelOutput']['generation'][-10:])


        if label == None:
            labels = None

        elif label == -1:
            labels = [1] * cot_len

        elif label >= 0 and label < cot_len:
            labels = [1] * label + [-1] * (cot_len - label)

        else: # bad label
            labels = None

        cot_data[q_id]['chain_of_thoughts'][cot_id]['eval'] = d['modelOutput']['generation']

        cot_data[q_id]['chain_of_thoughts'][cot_id]['labels'] = labels




    # processing to add counterfactual aug field to cot
    # Note: not used in final experiments so is empty (however training code expects this aug field)
    for d in cot_data:
        for cot in d['chain_of_thoughts']:
            cot['augs'] = []


    os.makedirs(os.path.dirname(args.outputdir), exist_ok=True)
    d_split = filename = os.path.splitext(os.path.basename(args.path))[0]
    with open(f'{args.outputdir}/{d_split}_autolabel.json', 'w') as f:

        json.dump(cot_data, f, indent=2)

