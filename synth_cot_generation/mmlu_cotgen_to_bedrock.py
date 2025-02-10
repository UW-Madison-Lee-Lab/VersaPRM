from collections import defaultdict
import json
import os
import argparse


def get_template(data):
    '''
    data is list of tuples, each tuple is question and corresponding CoT
    '''

    q_template = 'Given the following question and candidate answers, choose the best answer.\n{question}\n\nYour response should end with "The answer is (X)." where X is a letter from the provided choices.\nEach reasoning step in your response should be delimited by two newline characters\n\nLet\'s think step by step.'

    template = ''
    for q, cot in data:
        q_prompt = q_template.format(question=q)
        template += f'<|start_header_id|>user<|end_header_id|>\n\n{q_prompt}<|eot_id|>'
        template += f'<|start_header_id|>assistant<|end_header_id|>\n\n{cot}<|eot_id|>'

    template += f'<|start_header_id|>user<|end_header_id|>\n\n{q_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

    return template

def process_question(data):
    letters = 'ABCDEFGHIJKLMNOP'
    choice_str = ''
    for idx, c in enumerate(data['options']):
        choice_str += f'\n{letters[idx]}. {c}'

    question = 'Question: ' + data['question'] + choice_str
    return question

def get_mmlu_pro_fewshot_templates(fewshot_path):
    with open(fewshot_path, 'r') as f:
        mmlu_data = json.load(f)

    # key will be the category, value will be list of question, cot pairs
    fewshot_dict = defaultdict(list)
    
    for d in mmlu_data:

        fewshot_dict[d['category']].append((process_question(d), d['cot_content']))

    for subject in fewshot_dict:
        fewshot_dict[subject] = get_template(fewshot_dict[subject])

    return fewshot_dict

def add_llm_prompts(qa_data, fewshot_path):

    templates = get_mmlu_pro_fewshot_templates(fewshot_path)
    for qa in qa_data:
        template = templates[qa['category']]
  

        # very hacky way to add in question
        # as seems some parsing bug when trying to use `template.format(qa['question'])` due to math symbols
        qa['llm_prompt'] = template[:-274] + qa['question'] + template[-264:]





def convert_mmlu_to_bedrock_input(qa, temp=0.8, max_gen_len=2-48, num_gen=128):
    '''
    converts to list of dictionaries for bedrock input
    '''


    # for each ID, we add +idx, to ensure they are unique
    return [{'recordId': f'{qa['id']}+{i}',
                'modelInput':{'prompt':qa['llm_prompt'],
                        'temperature':temp,
                        'max_gen_len':max_gen_len}} for  i in range(num_gen)]


if __name__=='__main__':




    parser = argparse.ArgumentParser(description='Script to process mmlu questions into bedrock input file for CoT generation.')
    parser.add_argument('-p','--path', type=str, help='Path to mmlu questions', default='./data/mmlu_train_questions.json')
    parser.add_argument('-n','--samples', type=int, help='number of CoTs to generate per questions', default=16)
    parser.add_argument('-f','--fewshotpath', type=str, help='Path to json file with fewshot examples', default='./data/mmlu_val.json')
    parser.add_argument('-o','--outputdir', type=str, help='directory to store output bedrock file', default='bedrock_inputs')

    args = parser.parse_args()


    temp = 0.8
    max_gen_len= 2048

    with open(args.path, 'r') as f:
        qa_data = json.load(f)

    add_llm_prompts(qa_data, fewshot_path=args.fewshotpath)


    bedrock_inputs = []
    for qa in qa_data:
        bedrock_inputs.extend(convert_mmlu_to_bedrock_input(qa,
                                                            temp=temp,
                                                            max_gen_len=max_gen_len,
                                                            num_gen=args.samples))


    os.makedirs(args.outputdir, exist_ok=True)
    d_split = filename = os.path.splitext(os.path.basename(args.path))[0]
    with open(f'{args.outputdir}/{d_split}_{args.samples}_cotgen.jsonl', 'w') as f:
        for i in bedrock_inputs:
            json.dump(i, f)
            f.write('\n')