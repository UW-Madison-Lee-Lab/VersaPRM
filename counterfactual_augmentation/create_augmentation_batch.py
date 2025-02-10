"""
This script creates a counterfactual augmentation batch to be inputted to Llama 3.1 70B in AWS Bedrock.
To use this script, please change the following parameters below.
"""

import json
import random
import itertools
from tqdm import tqdm

# Change the following 3 parameters
input_file = "mmlu_labeled.json"
output_file = "mmlu_batch.jsonl"
use_math_error_types = False

random.seed(123456)
data = json.load(open(input_file, 'r'))
random.shuffle(data)

def get_all_correct_cot_idx(sample):
    idxs = []
    for i, cot in enumerate(sample['chain_of_thoughts']):
        if (cot.get('labels', None) == None):
            continue
        if (not -1 in cot['labels']):
            idxs.append(i)

    return idxs

error_type_list = [' - Conflicting Steps: The reasoning step includes statements that contradict previous steps.',
                  '  - Non-sequitur: The reasoning step introduces information that is irrelevant to the question.',
                  '  - Factual: The reasoning step contains incorrect statements of factual information.',
                  '  - False Assumption: The reasoning step makes an incorrect assumption about the question.',
                  '  - Contextual: The reasoning step misinterprets information given within the question/context']

if (use_math_error_types):
    error_type_list.append('  - Calculation: The reasoning step includes an incorrect computation of a mathematical expression or operation.')
    error_type_list.append('  - Incorrect Equation: The reasoning step includes an equation or formula that does not accurately represent the problem.')

error_perms = list(itertools.permutations(error_type_list))
random.shuffle(error_perms)

def make_round1_system_prompt(num):
    idx = num % len(error_type_list)
    curr_error_types = error_perms[idx][:2]

    round1_system_prompt = f"""You are a highly knowledgeable philosopher with expertise across many domains, tasked with analyzing reasoning processes. 
Your goal is to identify how a reasoning process could naturally deviate toward an incorrect conclusion through the introduction of subtle errors.

Here are a list of potential error types, all of which are equally valid:"""
   
    for err in curr_error_types:
        round1_system_prompt += f"\n{err}"
   
    round1_system_prompt += """\n\nInstructions:
You will be provided with:
1. A question.
2. A complete chain of reasoning steps, where each step is numbered (e.g., Step X).

Your task is to:
1. Identify the major factual information, reasoning, and conclusions within the reasoning steps
3. Explain how to generate an incorrect step to replace one of the existing steps. This should include:
   - Identifying a step where the reasoning could naturally deviate.
   - Speculating what type of error would be most appropriate to introduce at the chosen step.
4. Introduce an incorrect next step that aligns stylistically with the previous steps. This incorrect step should:
   - Reflect a deviation in reasoning that significantly harms the correctness.
   - Appear natural and believable in the context of the reasoning process.
5. Clearly explain how the incorrect step is an error, highlighting the specific logical or conceptual flaw.

Output Format:

STEP_SUMMARY:
[Summarize the reasoning within the steps in 1-2 sentences, identifying major information, logical steps, and conclusions.]

INCORRECT_STEP_GEN:
[Explain how the reasoning at a specific step could deviate naturally into being incorrect. Clearly describe the type of error that could be introduced at this step.]

ERROR_TYPE:
[The name of the type of error chosen to be introduced.]

STEP_NUM:
[The number of the step that was identified as a place where the reasoning could naturally deviate. Only include the number here.]

INCORRECT_STEP:
[Write the incorrect step in the same tone and style as the other steps. Wrap the incorrect step inside curly braces (e.g. {incorrect step}).]

ERROR_EXPLANATION:
[Explain how the incorrect step fits the definition of the selected error type, identifying the specific flaw.]"""
   
    return round1_system_prompt

def get_fewshot_prompt():
    QUESTIONS = ["Select the one true statement.\nA. In an animal cell, the endoplasmic reticulum helps ribosomes build proteins.\nB. The vacuoles of an animal cell use sunlight to make sugar.\nC. Plant cells can have a nucleus but do not have chloroplasts.",
                 "Cancer occurs most often in adults over what age?\nA. twenty\nB. fifty\nC. thirty\nD. seventy",
                 "Which type of sentence is this?\nAs Dirk sat down on the rickety old chair, it abruptly collapsed beneath him.\nA. simple\nB. compound\nC. complex\nD. compound-complex",
                 "Would you find the word crude on a dictionary page with the following guide words?\ncarton - cuff\nA. yes\nB. no"]
    
    cots = [[
          "Let's evaluate each statement:",
          "A. In an animal cell, the endoplasmic reticulum helps ribosomes build proteins. This statement is TRUE. The endoplasmic reticulum (ER) is a type of organelle that helps synthesize and transport proteins, often working in conjunction with ribosomes.",
          "B. The vacuoles of an animal cell use sunlight to make sugar. This statement is FALSE. Animal cells do not have chloroplasts, which are responsible for photosynthesis (converting sunlight into sugar). Vacuoles in animal cells are involved in storage and waste management, not photosynthesis.",
          "C. Plant cells can have a nucleus but do not have chloroplasts. This statement is FALSE. While it's true that plant cells have a nucleus, they also have chloroplasts, which are essential for photosynthesis. Chloroplasts are found in plant cells and are responsible for converting sunlight into sugar.",
          "The answer is A."
        ],
        [
          "To answer this question, let's consider the typical age range for cancer occurrence.",
          "Most types of cancer are more common in people over the age of 50, as the risk of developing cancer tends to increase with age. While some cancers, like leukemia, can occur in younger people, the majority of cancers are diagnosed in adults over 50.",
          "Given this information, the correct answer is:",
          "the answer is B."
        ],
        [
          "To determine the type of sentence, let's analyze its structure:",
          "1. The sentence has two independent clauses:\n   - \"Dirk sat down on the rickety old chair\"\n   - \"it abruptly collapsed beneath him\"",
          "2. These two independent clauses are joined by a comma, which indicates a dependent clause is not present.",
          "3. However, there is a dependent clause \"As Dirk sat down on the rickety old chair\" that provides additional information about when the main action took place.",
          "4. Since the sentence has one dependent clause and two independent clauses, it does not meet the criteria for a simple sentence (A), which has one independent clause. It also does not meet the criteria for a compound sentence (B), which has two independent clauses without a dependent clause.",
          "5. The sentence also does not meet the criteria for a compound-complex sentence (D), which has two or more independent clauses and one or more dependent clauses.",
          "6. Therefore, the sentence with one dependent clause and two independent clauses is a complex sentence (C).",
          "The answer is C."
        ],
        [
          "To determine if the word \"crude\" would be found on a dictionary page with the guide words \"carton - cuff\", we need to consider the alphabetical order of the words.",
          "The word \"crude\" starts with \"cr\", which falls between \"cart\" and \"cuff\" in alphabetical order. ",
          "Therefore, the word \"crude\" would be found on a dictionary page with the guide words \"carton - cuff\".",
          "The answer is A."
        ]]
    MODEL_RESPONSES = ["STEP_SUMMARY:\nThe reasoning process evaluates three statements regarding cell biology, considering the roles and components of animal and plant cells, particularly the involvement of the endoplasmic reticulum, vacuoles, and chloroplasts. The conclusion, based on the evaluations, identifies statement A as the true statement.\n\nINCORRECT_STEP_GEN:\nThe reasoning at Step 3 could deviate naturally into being incorrect by incorrectly classifying a fundamental characteristic unique to plant cells. A false assumption could be introduced at this step about the presence or absence of essential organelles in plant cells, thereby skewing the validity of the statement under evaluation.\n\nERROR_TYPE:\nFalse Assumption\n\nSTEP_NUM:\n3\n\nINCORRECT_STEP:\nStep 3: {C. Plant cells can have a nucleus but do not have chloroplasts. This statement is TRUE because a few plant cells, such as the hair cells found on the surface of plant roots, lack chloroplasts but still maintain a nucleus, which regulates cellular activity.}\n\nERROR_EXPLANATION:\nThis incorrect step fits the definition of a False Assumption error type. While acknowledging that plant cells generally contain chloroplasts for photosynthesis, it introduces an error by assuming that the absence of chloroplasts in a specific type of plant cell (though extremely rare) generalizes to all plant cells, thereby incorrectly validating statement C as true and undermining the overall conclusion by introducing ambiguity between statements A and C.",
                       "STEP_SUMMARY:\nThe reasoning steps analyze the common age range for cancer occurrence and conclude that most cancers are diagnosed in adults over 50, making option B the correct answer.\n\nINCORRECT_STEP_GEN:\nThe reasoning could naturally deviate into being incorrect at Step 1. This is because the statement about the majority of cancers being diagnosed in adults over 50 could be incorrectly qualified, allowing for a misinterpretation of the common age range for cancer occurrence.\n\nERROR_TYPE:\nContextual\n\nSTEP_NUM:\n1\n\nINCORRECT_STEP:\n{Most rare types of cancer are more common in people over the age of 50, as the risk of developing these specific cancers tends to increase with age. While most common types of cancers, like breast and skin cancers, can occur across all age groups, a few rare types are seen more often in adults over 50.}\n\nERROR_EXPLANATION:\nThis incorrect step fits the definition of a contextual error. Although the original Step 1 stated that most types of cancer are more common in people over 50, the revised step incorrectly qualifies this statement by introducing the distinction between rare and common types of cancer. This qualification creates a different implication: that it is only rare cancers that are more common in adults over 50, potentially misdirecting the conclusion and leading to incorrect inferences about the age most common for cancer occurrence.",
                       "STEP_SUMMARY:\nThe reasoning steps analyze the structure of the given sentence to determine its type, breaking it down into independent and dependent clauses and applying the definitions of simple, compound, complex, and compound-complex sentences. The conclusion drawn is that the sentence is complex due to having one dependent clause and multiple independent clauses.\n\nINCORRECT_STEP_GEN:\nThe reasoning at Step 2 could naturally deviate into being incorrect by introducing a False Assumption about the nature of the conjunction used to connect the clauses. This assumption could incorrectly interpret the role of the comma in the sentence, potentially changing the classification of the sentence entirely.\n\nERROR_TYPE:\nFalse Assumption\n\nSTEP_NUM:\n2\n\nINCORRECT_STEP:\n{2. These two independent clauses are joined by a comma, which suggests that the second clause \"it abruptly collapsed beneath him\" should be interpreted as a dependent clause since it immediately follows a comma and is related to a distinct subject.}\n\nERROR_EXPLANATION:\nThis incorrect step is an example of a False Assumption error because it misinterprets the function of the comma in connecting the two clauses. The error assumes that because the second independent clause follows a comma and has a distinct subject, it should be considered a dependent clause. However, logically and grammatically, the comma in this case actually separates the two independent clauses rather than introducing a dependent clause. This incorrect assumption would lead to a misunderstanding of the sentence's structure and its classification as a type of sentence.",
                       "STEP_SUMMARY:\nThe reasoning involves determining if the word \"crude\" would be found on a dictionary page between the guide words \"carton - cuff\" by considering alphabetical order. It concludes that \"crude\" falls between the guide words and can be found on this dictionary page.\n\nINCORRECT_STEP_GEN:\nThe reasoning at Step 2 could naturally deviate into being incorrect if it fails to accurately assess the alphabetical order or misplaced the positioning of \"crude\" relative to the guide words. A factual error could occur by misjudging the alphabetical sequence, leading to an incorrect conclusion about where \"crude\" would be located.\n\nERROR_TYPE:\nFactual\n\nSTEP_NUM:\n2\n\nINCORRECT_STEP:\n{Since the word \"crude\" starts with \"cr\" and the second letter is \"r\", but comes after \"cr\" words that have a \"u\" as their third letter, it would be after \"cuff\" in alphabetical order.}\n\nERROR_EXPLANATION:\nThe incorrect step is a factual error, as it introduces an incorrect understanding of how words are sequenced alphabetically, considering the entire word spelling rather than just the initial letters. This subtle mistake overlooking the normal rules of dictionary ordering could lead an individual to incorrectly determine where \"crude\" falls relative to the guide words \"carton - cuff\"."]

    if (use_math_error_types):
        QUESTIONS[0] = "A rural household uses a stove to boil water, heating 20kg of water from 20\u2103 to 100\u2103, completely burning 2kg of dry firewood. The specific heat capacity of water is 4.2\u00d710^3J/(kg\u00b7\u2103), and the calorific value of dry firewood is 1.2\u00d710^7J/kg. Q: What is the heat released by the complete combustion of the dry firewood?"
        cots[0] = ["1. We first calculate the heat absorbed by the water using the formula:\n[heat absorbed by the water]=[specific heat capacity of water]\u00d7[mass of water]\u00d7[temperature change]",
            "Using the values from the question we get ((4.2\u00d710^3) J/(kg\u00b7\u2103))\u00d7((20) kg)\u00d7((100) \u2103)-((4.2\u00d710^3) J/(kg\u00b7\u2103))\u00d7((20) kg)\u00d7((20) \u2103)=560000 J",
            "Thus the heat absorbed by the water is 560000 J",
            "2. We next calculate the heat released by the complete combustion of the dry firewood using the formula:\n[heat released by the complete combustion of the dry firewood]=[mass of dry firewood]\u00d7[calorific value of dry firewood]",
            "Plugging in the values from the question we get ((2) kg)\u00d7((1.2\u00d710^7) J/kg)=2.4\u00d710^7 J",
            "Thus the heat released by the complete combustion of the dry firewood is 2.4\u00d710^7 J",
            "The answer is \\boxed{2.4\u00d710^7 J}"]
        MODEL_RESPONSES[0] = "STEP_SUMMARY:\nThe reasoning involves calculating the heat released by the complete combustion of dry firewood, which is equal to the product of the mass of dry firewood and the calorific value of dry firewood. It uses the formula Q = mc\u0394T to calculate the heat absorbed by the water, but the actual answer depends on the calculated heat released by the dry firewood combustion, which is calculated by Q = mc.\n\nINCORRECT_STEP_GEN:\nThe reasoning at Step 4 could naturally deviate into being incorrect by incorrectly introducing an equation or using the values provided in the problem for the mass of dry firewood and the calorific value. \n\nERROR_TYPE:\nIncorrect Equation\n\nSTEP_NUM:\n4\n\nINCORRECT_STEP:\n{We next calculate the heat released by the complete combustion of the dry firewood using the formula: \n[heat released by the complete combustion of the dry firewood]=[mass of dry firewood]^2\u00d7[calorific value of dry firewood]}\n\nERROR_EXPLANATION:\nThis incorrect step is an example of an Incorrect Equation error because it introduces an incorrect formula to calculate the heat released by the dry firewood combustion. The original correct formula is based on the principle that the heat released by combustion is equal to the mass of fuel multiplied by the calorific value of fuel, not squared mass. Using the squared mass would significantly and inaccurately increase the calculated heat released by combustion."

    few_shot_prompt = ""
    for i, question in enumerate(QUESTIONS):
        prompt = generate_prompt_with_full_cot(question, cots[i])
        few_shot_prompt += gen_few_shot(prompt, MODEL_RESPONSES[i])
    
    return few_shot_prompt

def generate_prompt_with_full_cot(question, cot):
    prompt = f"QUESTION:\n{question}\n\n"

    prompt+="REASONING STEPS:"
    for i, cot_step in enumerate(cot):
      prompt += f"\nStep {i+1}: " + "{" + f"{cot_step}" + "}"

    return prompt

def gen_few_shot(user_prompt, assistant_prompt):
    return f'<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{assistant_prompt}<|eot_id|>'

# fixed combine prompt function
def combine_prompt(sys_prompt, user_prompt, few_shot_prompt=''):
    return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>{few_shot_prompt}<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

batch_inputs = []
counter = 0

for i, sample in tqdm(enumerate(data)):
    question = sample['question']
    cot_idxs = get_all_correct_cot_idx(sample)
    if (len(cot_idxs) == 0):
            continue
    for cot_idx in cot_idxs:
        for j in range(2):
            cot = sample['chain_of_thoughts'][cot_idx]['steps']
            prompt = generate_prompt_with_full_cot(question, cot)

            system_prompt = make_round1_system_prompt(counter)
            counter += 1

            full_prompt = combine_prompt(system_prompt, prompt, get_fewshot_prompt())

            batch_input = {'recordId': sample['id']+ '+' + sample['chain_of_thoughts'][cot_idx]['cot_id'], 'modelInput': {
                            "prompt": full_prompt,
                            "temperature": 0.4,
                            "max_gen_len": 2048
                        }}

            batch_inputs.append(batch_input)

with open(output_file, 'w') as outfile:
    for entry in batch_inputs:
        json.dump(entry, outfile)
        outfile.write('\n')