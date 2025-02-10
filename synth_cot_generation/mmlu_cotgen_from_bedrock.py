import json
import os
import argparse
from tqdm import tqdm
import re


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return None

    

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

def dataset_to_id_dict(ds):
    '''
    returns dictionary with idx for each id
    '''

    id_dict = {}

    for idx,d in enumerate(ds):
        
        # might as well also add in chain of thoughts key here (bad practice tho...)
        d['chain_of_thoughts'] = []
        id_dict[d['id']] = idx

    return id_dict


def parse_bedrock_id(recordId):

    id, cot_id = recordId.split('+')

    return id, cot_id

def parse_cot(cot_str, delimiter='\n\n'):
    '''
        splits cot string into list
        parses out the generated answer
    '''

    cot_splitted = cot_str.strip().split(delimiter)
    cot_splitted = [item.strip() for item in  cot_splitted]
    parsed_answer = extract_answer(cot_splitted[-1])
    return cot_splitted, parsed_answer





if __name__=='__main__':




    parser = argparse.ArgumentParser(description='Script to process bedrock MMLU CoT generation outputs.')
    parser.add_argument('-p','--path', type=str, help='Path to mmlu questions', default='./data/mmlu_train_questions.json')
    parser.add_argument('-b','--bedrockpath', type=str, help='path to directory of bedrock outputs we want to proccess. Shoud contain jsonl.out files', default='./bedrock_outputs/mmlu_train')
    parser.add_argument('-o','--outputdir', type=str, help='directory to store proccessed outputs', default='./data')

    args = parser.parse_args()

    with open(args.path, 'r') as f:
        qa_data = json.load(f)
    id_dict = dataset_to_id_dict(qa_data)

    bedrock_data = get_data(args.bedrockpath)


    for d in tqdm(bedrock_data):

        id, cot_id = parse_bedrock_id(d['recordId'])


        if 'modelOutput' not in d:
            continue

        stop_reason = d['modelOutput']['stop_reason']

        # we want to ignore bad CoT
        if stop_reason == 'length':
            continue

        steps, parsed_answer = parse_cot(d['modelOutput']['generation'])
        if parsed_answer == None: # again to ignore bad CoT
            continue


        idx = id_dict[id]
        qa_data[idx]['chain_of_thoughts'].append({'steps':steps, 
                                            'parsed_answer':parsed_answer,
                                            'parsed_answer_correctness': parsed_answer==qa_data[idx]['answer'],
                                            'cot_id':cot_id})
        


    os.makedirs(os.path.dirname(args.outputdir), exist_ok=True)
    d_split = filename = os.path.splitext(os.path.basename(args.path))[0]
    with open(f'{args.outputdir}/{d_split}_cot.json', 'w') as f:
        json.dump(qa_data, f, indent=2)

    

