import json
import argparse
import os

sys_prompt = '''You are an experienced evaluator specializing in assessing the quality of reasoning steps in problem-solving. Your task is to find the first BAD step in a student's solution to a multiple choice question.

You will judge steps as GOOD, OK or BAD based on the following criteria:
1. GOOD Step
A step is classified as GOOD if it meets all of these criteria:
- Correct: Everything stated is accurate and aligns with known principles or the given problem.
- Verifiable: The step can be verified using common knowledge, simple calculations, or a quick reference (e.g., recalling a basic theorem). If verifying requires extensive effort (e.g., detailed calculations or obscure references), mark it BAD instead.
- Appropriate: The step fits logically within the context of the preceding steps. If a prior mistake exists, a GOOD step can correct it.
- Insightful: The step demonstrates reasonable problem-solving direction. Even if ultimately progress in the wrong direction, it is acceptable as long as it represents a logical approach.

2. OK Step
A step is classified as OK if it is:
- Correct and Verifiable: Contains no errors and can be verified.
- Unnecessary or Redundant: Adds little value, such as restating prior information or providing basic encouragement (e.g., “Good job!”).
- Partially Progressing: Makes some progress toward the solution but lacks decisive or significant advancement.

3. BAD Step
A step is classified as BAD if it:
- Is Incorrect: Contains factual errors, misapplies concepts, derives an incorrect result, or contradicts the ground truth answer
- Is Hard to Verify: Requires significant effort to confirm due to poor explanation.
- Is Off-Topic: Includes irrelevant or nonsensical information.
- Derails: Leads to dead ends, circular reasoning, or unreasonable approaches.

#### Task Description
You will be provided with:
1. A Multiple Choice Question
2. A Ground Truth Answer
3. A Student's Step-by-Step Solution, where each step is enclosed with tags and indexed from 0.

Once you identify a BAD step, return the index of the earliest BAD step. Otherwise,
return the index of -1 (which denotes all steps are GOOD or OK).
Please put your final answer (i.e., the index) in \\boxed{}.
'''

template = '''
The following is a multiple choice question and its ground truth answer. You are also given a students solution (split into step, enclosed with tags and indexed from 0):

[Multiple Choice Question]
{question}

[Ground Truth Answer]
{answer}

[Student Solution]
{solution}
'''

def process_cot(cot):
    solution = ''
    for i, step in enumerate(cot):
        solution += f'<step_{i}>\n{step}\n</step_{i}>\n\n'

    return solution

def get_user_prompt(question, answer, steps):


    solution = process_cot(steps)


    return template.format(question=question, answer=answer, solution=solution)

if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Script to process MMLU questions and CoTs into into bedrock input file for autolabeling.')
    parser.add_argument('-p','--path', type=str, help='Path to mmlu questions with cot', default='./data/mmlu_train_questions_cot.json')
    parser.add_argument('-o','--outputdir', type=str, help='directory to store proccessed outputs', default='./bedrock_inputs')

    args = parser.parse_args()

    with open(args.path, 'r') as f:
        cot_data = json.load(f)


    temp = 0
    max_gen_len = 2048


    with open(args.path, 'r') as f:
        cot_data = json.load(f)

    bedrock_autolabels = []
    for d in cot_data:
        question = d['question']
        answer = d['answer']
        q_id = d['id']

        for cot in d['chain_of_thoughts']:
            user_prompt = get_user_prompt(question, answer, cot['steps'])
            prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

            cot_id = cot['cot_id']
            bedrock_autolabels.append({'recordId': f'{q_id}+{cot_id}',
                                    'modelInput':{'prompt':prompt,
                                    'temperature':temp,
                                    'max_gen_len':max_gen_len}})
        

    os.makedirs(os.path.dirname(args.outputdir), exist_ok=True)
    d_split = filename = os.path.splitext(os.path.basename(args.path))[0]
    with open(f'{args.outputdir}/{d_split}_autolabel.jsonl', 'w') as f:
        for d in bedrock_autolabels:
            json.dump(d, f)
            f.write('\n')