import torch
from torch import nn
from typing import Union, Tuple

from ...v7_goose.block.rwkv7_block_config_map import RWKV7BlockConfigMap

class Qwrky7BlockConfigMap(RWKV7BlockConfigMap):

    # RMS Norm eps
    rms_norm_eps: float = 1e-6
    # Attention QKV bias
    attention_bias: bool = True
    # Attention output bias
    attention_output_bias: bool = False

    def __init__(
        self, 
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = True,
        attention_output_bias: bool = False,
        **kargs
    ):
        '''
        Config with RMS Norm eps
        And alias for hidden_size_mlp
        '''
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_output_bias = attention_output_bias
        super().__init__(**kargs)

    
    def get_hidden_size_mlp(self) -> int:
        '''
        Intermidiate size of the MLP,
        Alias for get_hidden_size_ffn
        '''
        return self.get_hidden_size_ffn()

    # ---
    # Duplicator & Normalizer
    # ---

    def new_block_config_map(self, **kwargs) -> 'Qwrky7BlockConfigMap':
        '''
        Returns a new config map with updated values
        '''

        new_dict = {}
        for key in Qwrky7BlockConfigMap.__dataclass_fields__:
            if key in self.__dict__:
                new_dict[key] = self.__dict__[key]
        new_dict.update(kwargs)

        return Qwrky7BlockConfigMap(**new_dict)

    @staticmethod
    def normalize(config_map: any) -> 'Qwrky7BlockConfigMap':
        '''
        Converts either maps, objs or Qwrky7BlockConfigMap
        '''
        if isinstance(config_map, Qwrky7BlockConfigMap):
            return config_map
        
        dict_obj = None
        if isinstance(config_map, dict):
            dict_obj = config_map
        elif hasattr(config_map, '__dict__'):
            dict_obj = config_map.__dict__
        
        if dict_obj is not None:
            # Filter for only valeus in Qwrky7BlockConfigMap
            new_dict = {}
            for key, value in dict_obj.items():
                if key in Qwrky7BlockConfigMap.__dataclass_fields__:
                    new_dict[key] = value
            return Qwrky7BlockConfigMap(**new_dict)

        raise ValueError(f"Unsupported config_map type: {type(config_map)}")
