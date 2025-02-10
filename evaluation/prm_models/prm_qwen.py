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


def check_model_precision(model):
    for name, param in model.named_parameters():
        print(f'Parameter: {name}, dtype: {param.dtype}')


def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer


class QwenPRM(PRM):

    def __init__(
        self,
        aggregation: str = 'full',
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        model_id: str = 'UW-Madison-Lee-Lab/Qwen-PRM800K'
    ) -> None:
        self.device = (
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.quantization_config = quantization_config
        self.model_id = model_id

        self.tokenizer = get_tokenizer(self.model_id)

        self.candidate_tokens = [12, 10]

        if self.quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        if not quantization_config:
            self.model.to(self.device)

        self.aggregation = aggregation


    def __call_single(self, single_beam: str) -> list[float]:
        '''
        Computes scores for each reasoning step in the single_beam.

        Args:
            single_beam (str): A single reasoning beam, consisting of Question + Solution.

        Returns:
            list[float]: The scores for each step in the Solution.
        '''
        # Tokenize the entire input
        encoded = self.tokenizer.encode(single_beam, return_tensors='pt').to(self.device)
        input_ids = encoded[0]
        total_length = input_ids.size(0)

        input_id = torch.tensor([self.tokenizer.encode(single_beam)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            # print(logits)
            scores = logits.softmax(dim=-1)[:,:,1] 
            # print(scores)
            step_scores = scores[input_id == 22701] # step tag is 22701
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
        Computes scores for a list of reasoning beams.

        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of StepScore objects, each containing step and score.
        '''
        result = []

        for beam in steps:
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result