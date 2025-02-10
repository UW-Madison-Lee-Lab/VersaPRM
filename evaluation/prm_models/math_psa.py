import math
import statistics
from typing import Optional
from tqdm import tqdm
import torch
from peft import PeftModel
from .prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import json

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class MathPSA(PRM):
    def __init__(
        self,
        aggregation: str = 'full', #the way how prm step scores will be aggregated in a solution
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        model_id: str = 'Qwen/Qwen2.5-Math-7B-Instruct',
        downloaded_adapter_path: str = None, # local path to the downloader adapter weights
    ) -> None:
        self.device = (
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n' #ки
        self.step_tag2 = '\n\n'

        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,add_eos_token=False)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16).to(self.device)
        
        self.adapter_path = downloaded_adapter_path

        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        if not quantization_config:
            self.model.to(self.device)
        self.aggregation = aggregation


        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = 'left'  # Allow batched inference

        self.candidate_tokens = self.tokenizer.encode(f' {self.good_token} {self.bad_token}') # [488, 481]
        self.step_tag_id = self.tokenizer.encode(f' {self.step_tag}')[-1] # 76325



    def __call_single(self, single_beam: str) -> float | list[float]:
        input_for_prm = single_beam

        ###
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            # print(logits)
            scores = logits.softmax(dim=-1)[:,:,0] 
            # print(scores)
            step_scores = scores[input_id == self.step_tag_id]
            step_probs  = step_scores.tolist()
        ###

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