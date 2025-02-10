import torch
from torch import Tensor
from typing import Union, Tuple

from ...v7_goose.block.rwkv7_time_mix import RWKV7TimeMix
from ...v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Qwrky7TimeMix(RWKV7TimeMix):
    def __init__(self, configMap: Union[RWKV7BlockConfigMap, any]):
        super().__init__(configMap)

    def forward(
        self, 
        x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, v_first_val:Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        # Apply qwen2 rotary embeddings
        # see: https://github.com/huggingface/transformers/blob/6b550462139655d488d4c663086a63e98713c6b9/src/transformers/models/qwen2/modeling_qwen2.py#L68
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            x = (x * cos) + (rotate_half(x) * sin)

        # Apply time mixing as usual
        return super().forward(x, shift_state_in, wkv_state_in, v_first_val)