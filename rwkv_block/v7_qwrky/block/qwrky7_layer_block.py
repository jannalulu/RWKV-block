import torch
from torch import nn
from typing import Union, Tuple

from ...v7_goose.block.rwkv7_layer_block import RWKV7LayerBlock
from ...v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap
from .qwrky7_time_mix import Qwrky7TimeMix
from .qwrky7_block_config_map import Qwrky7BlockConfigMap

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP

class Qwrky7LayerBlock(torch.nn.Module):
    def __init__(
            self, 
            configMap: Union[Qwrky7BlockConfigMap, RWKV7BlockConfigMap, any]
        ):

        # The configMap to use
        configMap = Qwrky7BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        device = configMap.get_device(None)
        rms_norm_eps = configMap.rms_norm_eps

        # Setup the modules
        self.input_layernorm = Qwen2RMSNorm(hidden_size, rms_norm_eps).to(device)
        self.self_attn = Qwrky7TimeMix(configMap)

        self.post_attention_layernorm = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps).to(device)
        self.mlp = Qwen2MLP({
            "hidden_size": hidden_size,
            "intermediate_size": configMap.get_hidden_size_ffn()
        }).to(device)

        # Setup droupout at block level
        dropout_rate = configMap.dropout_rate
        if dropout_rate > 0.0:            
            self.drop0 = nn.Dropout(p = dropout_rate,device=device)
            self.drop1 = nn.Dropout(p = dropout_rate,device=device)
        else:
            self.drop0 = nn.Identity(device=device)
            self.drop1 = nn.Identity(device=device)
    
    def reset_parameters(self):
        '''
        Reset the parameters of the block, to an initial state used for training a model from scratch
        '''
        # Call the sub blocks reset_parameters
        self.self_attn.reset_parameters()

        # Reset the layernorms
        self.input_layernorm.weight.data.fill_(1.0)
        self.post_attention_layernorm.weight.data.fill_(1.0)
        
        # Update the linear layers
        self.mlp.gate_proj.reset_parameters()
        self.mlp.up_proj.reset_parameters()
        self.mlp.down_proj.reset_parameters()


    def forward(
        self, 
        x:torch.Tensor, # hidden state
        last_state: tuple[torch.Tensor,torch.Tensor], 
        v_first:torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor,tuple[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor]:
        '''
        Forward the block given the input tokens and the last state, and position embeddings
        Last state is a tuple of the following
        - time mix shift state
        - time mix wkv state

        Returns a pair of the output embedding, v_first and the next state
        '''

        # Ensure everything is in the right device
        x = x.to(self.input_layernorm.weight.device)

        # Forward the time mix, with position embeddings
        att_out, tmix_shift, tmix_wkv, v_first = self.att(
            self.input_layernorm(x),
            last_state[0], # tmix_shift,
            last_state[1], # tmix_wkv
            v_first,
            position_embeddings=position_embeddings
        )

        # x = x + att_out
        x = self.drop0(x + att_out)

        mlp_out, mlp_state = self.mlp(
            self.post_attention_layernorm(x),
            last_state[2] # cmix_shift,
        )

        # x = x + ffn_out
        x = self.drop1(x + mlp_out)

        # Return the output
        return x, (tmix_shift, tmix_wkv), v_first
    