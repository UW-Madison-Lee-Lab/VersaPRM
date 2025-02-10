import math
import statistics
import json
import torch
from torch.types import Device
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .prm_interface import PRM, StepScore

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class MathShepherd(PRM):
    def __init__(
        self,
        aggregation: str = 'full',#the way how prm step scores will be aggregated in a solution
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        model_id: str = 'peiyi9979/math-shepherd-mistral-7b-prm'
    ) -> None:
        self.device = (
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.good_token = '+' #token in the vocabulary set that indicates the probability of good step 
        self.bad_token = '-' #token in the vocabulary set that indicates the probability of bad step 
        self.step_tag = 'ки'# step deliminator to locate the logits to get the prm score for each step

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        
        self.candidate_tokens = self.tokenizer.encode(f'{self.good_token} {self.bad_token}')[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f'{self.step_tag}')[-1] # 12902
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                       quantization_config=quantization_config).eval()
        if not quantization_config:
            self.model.to(self.device)

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> float | list[float]:
        input_for_prm = single_beam
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]#for the good token
            step_scores = scores[input_id == self.step_tag_id]
       
        step_probs  = step_scores.tolist()

        if self.aggregation == 'min':
            return min(step_probs)
        elif self.aggregation == 'max':
            return max(step_probs)
        elif self.aggregation == 'mean':
            return statistics.mean(step_probs)
        elif self.aggregation == 'prod':
            return math.prod(step_probs)
        elif self.aggregation == 'last':
            return step_probs[-1]
        elif self.aggregation == 'full':
            return step_probs
        else:
            raise NotImplementedError

    def __call__(self, steps: list[str]) -> list[StepScore]:
        '''
        Args:
            steps (list[str]): A list of reasoning solutions.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        '''

        result = []

        for beam in steps:#each beam is a cot solution, each step_score is a list of prm step scores for this solution(if aggregation methods is full)
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result