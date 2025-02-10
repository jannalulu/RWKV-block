import torch
from typing import Union, Tuple

from ...v7_goose.block.rwkv7_layer_block import RWKV7LayerBlock
from ...v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap
from .qwrky7_time_mix import Qwrky7TimeMix

class Qwrky7LayerBlock(RWKV7LayerBlock):
    def __init__(self, configMap: Union[RWKV7BlockConfigMap, any]):
        self.att = Qwrky7TimeMix(configMap)
        super().__init__(configMap)

    
    def forward(
        self, x:torch.Tensor,
        last_state: tuple[torch.Tensor,torch.Tensor,torch.Tensor], 
        v_first:torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Forward the block given the input tokens and the last state, and position embeddings
        Last state is a tuple of the following
        - time mix shift state
        - time mix wkv state
        - channel mix shift state

        Returns a pair of the output embedding, v_first and the next state
        '''

        # # Ensure everything is in the right device
        # x = x.to(self.ln1.weight.device)
        # last_state = [ s.to(self.ln1.weight.device) for s in last_state ]

        # Note, that this only applies for layer 0
        ln0_out = self.ln0(x)

        # Forward the time mix, with position embeddings
        att_out, tmix_shift, tmix_wkv, v_first = self.att(
            self.ln1(ln0_out),
            last_state[0], # tmix_shift,
            last_state[1], # tmix_wkv
            v_first,
            position_embeddings=position_embeddings
        )

        # x = x + att_out
        x = self.drop0(ln0_out + att_out)

        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state[2] # cmix_shift,
        )

        # x = x + ffn_out
        x = self.drop1(x + ffn_out)

        # Return the output
        return x, (tmix_shift, tmix_wkv, ffn_state), v_first
    