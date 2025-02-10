from typing import Optional

import torch
from torch import Tensor
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from interfaces import DEBUG, BaseGenerator
from key_value_utils import (
    filter_past_key_values,
    left_pad,
    move_past_key_values,
    stack_past_key_values,
    stack_sequences,
)

SYSTEM_MESSAGE = """
Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
"""


class LlamaGenerator(BaseGenerator):
    def __init__(
        self,
        max_new_tokens: int = 1000,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        hf_token: Optional[str] = None,
        use_past_key_values: bool = True,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        secondary_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.use_past_key_values = use_past_key_values
        self.batch_size = batch_size
        self.device = device
        self.secondary_device = secondary_device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            quantization_config=quantization_config,
        )

        if quantization_config is None:
            self.model = self.model.to(self.device, dtype=dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        self.step_ids = [
            i
            for t, i in self.tokenizer.vocab.items()  # type: ignore
            if "\n\n" in t or "ĊĊ" in t
        ]
        self.step_ids += self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)

        self.eos_id = self.tokenizer.encode(
            "<|end_of_text|>", add_special_tokens=False
        )[0]
        self.pad_id = 128248
        self.pad_token = self.tokenizer.decode(self.pad_id)
        self.model.generation_config.pad_token_id = self.pad_id
        self.model.generation_config.eos_token_id = self.step_ids + [
            self.eos_id,
            self.model.generation_config.eos_token_id,
        ]
        self.max_new_tokens = max_new_tokens

    def encode(self, question: str) -> Tensor:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        return input_ids.to(self.secondary_device)  # type: ignore

    def decode(self, input_ids: Tensor) -> str:
        return self.tokenizer.decode(input_ids).replace(self.pad_token, "")

    def is_complete(self, input_ids: Tensor) -> Tensor:
        decoded_beams = [self.decode(seq) for seq in input_ids]
        user_tag = "<|start_header_id|>user<|end_header_id|>"
        is_complete = ["\\boxed" in s[s.find(user_tag) :] for s in decoded_beams]
        return torch.tensor(is_complete)

    def init_state(
        self,
        input_ids: Tensor,
    ) -> Optional[tuple[tuple[Tensor, Tensor], ...]]:
        if not self.use_past_key_values:
            return None

        batched_past_key_values = []

        for i in range(0, input_ids.shape[0], self.batch_size):
            batched_input_ids = input_ids[i : i + self.batch_size].to(self.device)
            attention_mask = (batched_input_ids != self.pad_id).long()
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batched_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    return_legacy_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            assert past_key_values is not None
            batched_past_key_values.append(
                tuple(
                    (
                        k[:, :, :-1].to(self.secondary_device),
                        v[:, :, :-1].to(self.secondary_device),
                    )
                    for k, v in past_key_values
                )
            )

        return stack_past_key_values(batched_past_key_values)

    def filter_state(
        self,
        state: Optional[tuple[tuple[Tensor, Tensor], ...]],
        idxs: list[int],
    ) -> Optional[tuple[tuple[Tensor, Tensor], ...]]:
        if state is None:
            return None

        return filter_past_key_values(state, idxs)

    def inflate_state(
        self,
        state: Optional[tuple[tuple[Tensor, Tensor], ...]],
        n: int,
    ) -> Optional[tuple[tuple[Tensor, Tensor], ...]]:
        if state is None:
            return None

        past_key_values = state
        return tuple(
            (k.repeat(n, *([1] * (k.ndim - 1))), v.repeat(n, *([1] * (v.ndim - 1))))
            for k, v in past_key_values
        )

    def generate_statefull(
        self,
        input_ids: Tensor,
        past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        input_ids, past_key_values = left_pad(input_ids, past_key_values, self.pad_id)

        output_sequences = []
        output_past_key_values = []

        for i in range(0, input_ids.shape[0], self.batch_size):
            batched_input_ids = input_ids[i : i + self.batch_size].to(self.device)
            attention_mask = (batched_input_ids != self.pad_id).long()

            if past_key_values is None:
                batched_past_key_values = None
            else:
                batched_past_key_values = self.filter_state(
                    past_key_values,
                    list(range(i, min(i + self.batch_size, input_ids.shape[0]))),
                )
                assert batched_past_key_values is not None
                batched_past_key_values = move_past_key_values(
                    batched_past_key_values,
                    self.device,
                )

            outputs = self.model.generate(
                input_ids=batched_input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.8,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                eos_token_id=self.step_ids + [self.eos_id],
                pad_token_id=self.pad_id,
                use_cache=True,
                return_legacy_cache=True,
                return_dict_in_generate=True,
                past_key_values=batched_past_key_values,
            )

            if DEBUG:
                deterministic_outputs = self.model.generate(
                    input_ids=batched_input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=self.max_new_tokens,
                    max_length=None,
                    eos_token_id=self.step_ids + [self.eos_id],
                    pad_token_id=self.pad_id,
                    use_cache=True,
                    past_key_values=batched_past_key_values,
                )

                for i in range(len(batched_input_ids)):
                    idxs = attention_mask[i].bool()
                    outputs_control = self.model.generate(
                        input_ids=batched_input_ids[None, i, idxs],
                        attention_mask=attention_mask[None, i, idxs],
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        max_new_tokens=self.max_new_tokens,
                        max_length=None,
                        eos_token_id=self.step_ids + [self.eos_id],
                        pad_token_id=self.pad_id,
                        use_cache=True,
                    )
                    j = (~idxs).long().sum()
                    k = j + outputs_control.shape[1]
                    assert torch.all(
                        outputs_control == deterministic_outputs[None, i, j:k]
                    )

            output_sequences.append(outputs["sequences"].to(self.secondary_device))
            output_past_key_values.append(
                move_past_key_values(outputs["past_key_values"], self.secondary_device)
            )

        return (
            stack_sequences(output_sequences, pad_id=self.pad_id),
            stack_past_key_values(output_past_key_values),
        )

    def generate_stateless(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, None]:
        input_ids, _ = left_pad(input_ids, None, self.pad_id)

        output_sequences = []

        for i in range(0, input_ids.shape[0], self.batch_size):
            batched_input_ids = input_ids[i : i + self.batch_size].to(self.device)
            attention_mask = (batched_input_ids != self.pad_id).long()

            outputs = self.model.generate(
                input_ids=batched_input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                eos_token_id=self.step_ids + [self.eos_id],
                pad_token_id=self.pad_id,
                use_cache=True,
                return_dict_in_generate=True,
            )

            output_sequences.append(outputs["sequences"].to(self.secondary_device))

        return stack_sequences(output_sequences, pad_id=self.pad_id), None

    def __call__(
        self,
        input_ids: Tensor,
        past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None,
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]:
        if self.use_past_key_values:
            return self.generate_statefull(input_ids, past_key_values)
        else:
            return self.generate_stateless(input_ids)
