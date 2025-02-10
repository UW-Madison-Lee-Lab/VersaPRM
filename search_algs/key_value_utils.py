from typing import Optional, overload

import torch
from torch import Tensor


@overload
def left_pad(
    input_ids: Tensor,
    past_key_values: tuple[tuple[Tensor, Tensor], ...],
    pad_id: int = 0,
) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]: ...


@overload
def left_pad(
    input_ids: Tensor,
    past_key_values: None,
    pad_id: int = 0,
) -> tuple[Tensor, None]: ...


def left_pad(
    input_ids: Tensor,
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]],
    pad_id: int = 0,
) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]:
    # realign sequences
    pad_idxs = input_ids == pad_id
    realigned_input_ids = []

    for i, seq in enumerate(input_ids):
        realigned_input_ids.append(torch.cat((seq[pad_idxs[i]], seq[~pad_idxs[i]])))

    if past_key_values is None:
        return torch.stack(realigned_input_ids), None

    # realign kv-cache
    if input_ids.shape[1] == past_key_values[0][0].shape[2]:
        from_generate = False
    else:
        assert input_ids.shape[1] == past_key_values[0][0].shape[2] + 1
        from_generate = True

    realigned_past_key_values = []

    for k, v in past_key_values:
        realigned_key, realigned_value = [], []

        # realign key and value for each batch
        for i, seq in enumerate(input_ids):
            kv_pad_idxs = (
                pad_idxs[i, :-1] | pad_idxs[i, 1:] if from_generate else pad_idxs[i]
            )
            realigned_key.append(
                torch.cat((k[i, :, kv_pad_idxs], k[i, :, ~kv_pad_idxs]), dim=1)
            )
            realigned_value.append(
                torch.cat((v[i, :, kv_pad_idxs], v[i, :, ~kv_pad_idxs]), dim=1)
            )

        realigned_past_key_values.append(
            (torch.stack(realigned_key), torch.stack(realigned_value))
        )

    return torch.stack(realigned_input_ids), tuple(realigned_past_key_values)


def stack_sequences(batched_ids: list[Tensor], pad_id: int = 0) -> Tensor:
    n_batches = len(batched_ids)
    seq_len = max(x.shape[1] for x in batched_ids)
    padded_ids = []

    for i in range(n_batches):
        ids = batched_ids[i]
        if ids.shape[1] < seq_len:
            padding = pad_id * torch.ones(
                ids.shape[0],
                seq_len - ids.shape[1],
                dtype=ids.dtype,
                device=ids.device,
            )
            ids = torch.cat((ids, padding), dim=1)

        padded_ids.append(ids)

    return torch.cat(padded_ids)


def stack_past_key_values(
    batched_past_key_values: list[tuple[tuple[Tensor, Tensor], ...]],
    pad_id: int = 0,
) -> tuple[tuple[Tensor, Tensor], ...]:
    n_batches = len(batched_past_key_values)
    n_layers = len(batched_past_key_values[0])
    seq_len = max(x[0][0].shape[2] for x in batched_past_key_values)
    stacked_past_key_values = []
    for _ in range(n_layers):
        j = 0
        ks = []
        vs = []
        for i in range(n_batches):
            k, v = batched_past_key_values[i][j]
            if k.shape[2] < seq_len:
                padding = pad_id * torch.ones(
                    k.shape[0],
                    k.shape[1],
                    seq_len - k.shape[2],
                    k.shape[3],
                    dtype=k.dtype,
                    device=k.device,
                )
                k = torch.cat((k, padding), dim=2)
                v = torch.cat((v, padding), dim=2)

            ks.append(k)
            vs.append(v)

            batched_past_key_values[i] = batched_past_key_values[i][j+1:]
            
        stacked_past_key_values.append((torch.cat(ks), torch.cat(vs)))

    return tuple(stacked_past_key_values)


def filter_past_key_values(
    past_key_values: tuple[tuple[Tensor, Tensor], ...],
    idxs: list[int],
) -> tuple[tuple[Tensor, Tensor], ...]:
    return tuple((k[idxs], v[idxs]) for k, v in past_key_values)


def move_past_key_values(
    past_key_values: tuple[tuple[Tensor, Tensor], ...],
    device: str,
) -> tuple[tuple[Tensor, Tensor], ...]:
    return tuple((k.to(device), v.to(device)) for k, v in past_key_values)
