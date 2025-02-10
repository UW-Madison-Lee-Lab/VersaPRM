from typing import Optional

import torch
from torch import Tensor
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from interfaces import DEBUG, LOGGING, BasePRM
from key_value_utils import (
    filter_past_key_values,
    left_pad,
    move_past_key_values,
    stack_past_key_values,
)


class GeneralPRM(BasePRM):
    def __init__(
        self,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        model_name: str = "Daewon0808/prm800k_qwen_fulltune",
        positive_tag: str = "+",
        negative_tag: str = "-",
        score_token: str = " \n\n\n\n",
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
            quantization_config=quantization_config,
            token=hf_token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        if quantization_config is None:
            self.model = self.model.to(self.device, dtype=dtype)

        plus_tag_id = self.tokenizer.encode(positive_tag)[-1]
        minus_tag_id = self.tokenizer.encode(negative_tag)[-1]
        self.candidate_ids = [plus_tag_id, minus_tag_id]
        self.pad_id: int = self.tokenizer.eos_token_id  # type: ignore
        self.pad_token = self.tokenizer.decode(self.pad_id)
        self.score_token = score_token
        self.score_id = self.tokenizer.encode(score_token)[-1]
        self.step_tokens = self.encode(score_token)
        self.tokenizer.pad_id = self.pad_id

    def encode(self, text: str | list[str]) -> Tensor:
        # TODO: Find better way to handle different padding tokens
        if isinstance(text, list):
            text = [
                s.replace(
                    "<|reserved_special_token_240|>",
                    self.pad_token,
                ).replace(
                    "<|eot_id|>",
                    self.score_token,
                )
                for s in text
            ]
        else:
            text = text.replace(
                "<|reserved_special_token_240|>",
                self.pad_token,
            ).replace(
                "<|eot_id|>",
                self.score_token,
            )

        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        if isinstance(text, list):
            max_len = max(len(i) for i in ids)  # type: ignore
            for i in range(len(ids)):  # type: ignore
                ids[i] += [self.pad_id] * (max_len - len(ids[i]))  # type: ignore

        return torch.tensor(ids, device=self.secondary_device)

    def compute_scores(
        self,
        input_ids: Tensor,
        logits: Tensor,
    ) -> Tensor:
        scores = torch.empty(len(input_ids), device=input_ids.device)
        for i in range(len(input_ids)):
            probs = logits[i, input_ids[i] == self.score_id].softmax(dim=-1)[:, 0]
            scores[i] = torch.min(probs)

        return scores

    def init_state(
        self, question: str
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]:
        input_ids = self.encode([f"{question}\n\n"])
        batched_past_key_values = []

        for i in range(0, input_ids.shape[0], self.batch_size):
            batched_new_input_ids = input_ids[i : i + self.batch_size].to(self.device)
            attention_mask = (batched_new_input_ids != self.pad_id).long()
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batched_new_input_ids,
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
                    (k.to(self.secondary_device), v.to(self.secondary_device))
                    for k, v in past_key_values
                )
            )

        return input_ids, stack_past_key_values(batched_past_key_values)

    def filter_state(
        self,
        state: tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]],
        idxs: list[int],
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]:
        input_ids, past_key_values = state
        input_ids = input_ids[idxs]

        if past_key_values is not None:
            past_key_values = filter_past_key_values(past_key_values, idxs)

        return input_ids, past_key_values

    def inflate_state(
        self, state: tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]], n: int
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]:
        input_ids, past_key_values = state
        input_ids = input_ids.repeat(n, 1)

        if past_key_values is not None:
            past_key_values = tuple(
                (k.repeat(n, *([1] * (k.ndim - 1))), v.repeat(n, *([1] * (v.ndim - 1))))
                for k, v in past_key_values
            )

        return input_ids, past_key_values

    def call_with_kv_cache(
        self,
        new_input_ids: Tensor,
        state: Optional[tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]],
    ) -> tuple[Tensor, tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]]:
        device = self.device
        if state is None:
            full_input_ids = new_input_ids
            past_key_values = None
            cache_position = None
        else:
            input_ids, past_key_values = state
            full_input_ids = torch.cat((input_ids, new_input_ids), dim=1)
            cache_position = torch.arange(
                input_ids.shape[1],
                full_input_ids.shape[1],
                device=device,
            )

        attention_mask = (full_input_ids != self.pad_id).long()
        full_position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        position_ids = full_position_ids[:, -new_input_ids.shape[1] :]

        output_scores = []
        output_past_key_values = []

        for i in range(0, new_input_ids.shape[0], self.batch_size):
            if past_key_values is None:
                batched_past_key_values = None
            else:
                batched_past_key_values = filter_past_key_values(
                    past_key_values,
                    list(range(i, min(i + self.batch_size, new_input_ids.shape[0]))),
                )
                batched_past_key_values = move_past_key_values(
                    batched_past_key_values,
                    device,
                )

            with torch.no_grad():
                outputs = self.model(
                    input_ids=new_input_ids[i : i + self.batch_size].to(device),
                    attention_mask=attention_mask[i : i + self.batch_size].to(device),
                    position_ids=position_ids[i : i + self.batch_size].to(device),
                    past_key_values=batched_past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    return_legacy_cache=True,
                    return_dict=True,
                )

                if DEBUG:
                    outputs_control = self.model(
                        input_ids=full_input_ids[i : i + self.batch_size].to(device),
                        attention_mask=attention_mask[i : i + self.batch_size].to(
                            device
                        ),
                        position_ids=full_position_ids[i : i + self.batch_size].to(
                            device
                        ),
                        use_cache=False,
                        return_dict=True,
                    )

                    for j in range(outputs.logits.shape[0]):
                        idxs = attention_mask[i + j, -new_input_ids.shape[1] :].bool()
                        assert torch.allclose(
                            outputs.logits[i][idxs],
                            outputs_control.logits[i, -new_input_ids.shape[1] :][idxs],
                            atol=1e-2,
                            rtol=1e-2,
                        )

            logits = outputs.logits[:, :, self.candidate_ids].to(self.secondary_device)
            output_scores.append(
                self.compute_scores(new_input_ids[i : i + self.batch_size], logits)
            )
            output_past_key_values.append(
                move_past_key_values(
                    outputs.past_key_values,
                    self.secondary_device,
                )
            )

        scores = torch.cat(output_scores)
        full_input_ids, past_key_values = left_pad(
            full_input_ids,
            stack_past_key_values(output_past_key_values),
            self.pad_id,
        )
        return scores, (full_input_ids, past_key_values)

    def call_without_kv_cache(
        self,
        new_input_ids: Tensor,
        state: Optional[tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]],
    ) -> tuple[Tensor, tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]]:
        device = self.device
        if state is None:
            full_input_ids = new_input_ids
        else:
            input_ids, _ = state
            full_input_ids = torch.cat((input_ids, new_input_ids), dim=1)

        attention_mask = (full_input_ids != self.pad_id).long()
        full_position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        output_scores = []

        for i in range(0, full_position_ids.shape[0], self.batch_size):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_input_ids[i : i + self.batch_size].to(device),
                    attention_mask=attention_mask[i : i + self.batch_size].to(device),
                    position_ids=full_position_ids[i : i + self.batch_size].to(device),
                    use_cache=False,
                    return_dict=True,
                )

            logits = outputs.logits[:, :, self.candidate_ids].to(self.secondary_device)
            output_scores.append(
                self.compute_scores(full_input_ids[i : i + self.batch_size], logits)
            )

        scores = torch.cat(output_scores)
        full_input_ids, _ = left_pad(
            full_input_ids,
            None,
            self.pad_id,
        )

        return scores, (full_input_ids, None)

    def __call__(
        self,
        new_text: list[str],
        state: Optional[
            tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]
        ] = None,
    ) -> tuple[Tensor, tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]]:
        new_input_ids_list = []

        for seq in self.encode(new_text):
            pad_idxs = seq == self.pad_id
            new_input_ids_list.append(
                torch.cat((seq[~pad_idxs][:-1], self.step_tokens, seq[pad_idxs]))
            )

        new_input_ids = torch.stack(new_input_ids_list)

        if self.use_past_key_values:
            scores, new_state = self.call_with_kv_cache(new_input_ids, state)

        else:
            scores, new_state = self.call_without_kv_cache(new_input_ids, state)

        if LOGGING:
            print("=" * 150)
            for sequence, score in zip(
                self.tokenizer.batch_decode(new_state[0]),
                scores,
            ):
                print(sequence)
                print("-" * 150)
                print(f"score: {score.item():.2f}\n")
                print("-" * 150)

        return scores, new_state
