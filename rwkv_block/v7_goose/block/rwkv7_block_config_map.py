from dataclasses import dataclass
from typing import Optional
from typing import Union
import torch

@dataclass
class RWKV7BlockConfigMap:

    """Configuration map for RWKV based models"""
    # Key properties for the block / model
    num_hidden_layers: int
    hidden_size: int

    head_size: int = 64

    # Dropout rate, should only be used in training
    dropout_rate: float = 0.0

    # Implementation backend to use
    tmix_backend: str = "auto"

    # ---
    # OPTIONAL PROPS
    #
    # Optional properties which can be derived
    # or can be overwritten by the user
    # ---

    # Current layer_id of the block
    layer_id: Optional[int] = None

    # Device and Data type
    device: Union[torch.device, str, None] = None
    dtype: Union[torch.dtype, str, None] = None

    # Channel mix / FFN block dimension size
    hidden_size_ffn: Optional[int] = None
    hidden_size_att: Optional[int] = None

    # # number of heads
    # n_head: Optional[int] = None

    # ---
    # Initializer, with excess arg ignore
    # ---
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        head_size: int = 64,
        dropout_rate: float = 0.0,
        tmix_backend: str = "auto",
        layer_id: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: Union[torch.dtype, str, None] = None,
        hidden_size_ffn: Optional[int] = None,
        hidden_size_att: Optional[int] = None,
        **kwargs
    ) -> None:
        '''
        Constructor for the config
        '''
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.tmix_backend = tmix_backend
        self.layer_id = layer_id
        self.device = device
        self.dtype = dtype
        self.hidden_size_ffn = hidden_size_ffn
        self.hidden_size_att = hidden_size_att

    # ---
    # OPTIONAL PROPS FETCHER
    # ---

    def get_layer_id(self, fallback:int) -> int:
        '''
        Returns the layer id
        '''
        if self.layer_id is not None:
            return self.layer_id
        return fallback
    
    def get_device(self, fallback:str) -> torch.device:
        '''
        Returns the device
        '''
        if self.device is not None:
            return torch.device(self.device)
        if fallback is not None:
            return torch.device(fallback)
        return torch.get_default_device()
    
    def get_dtype(self, fallback:str) -> torch.dtype:
        '''
        Returns the dtype
        '''
        if self.dtype is not None:
            key = self.dtype
        else:
            key = fallback

        # if dtype is already torch.dtype
        if isinstance(key, torch.dtype):
            return key
        
        # Get and Check if the dtype is instance of torch.dtype
        ret = getattr(torch, key) 
        assert isinstance(ret, torch.dtype), f"Invalid dtype: {self.dtype}"
        return ret
    
    # ---
    
    def get_hidden_size_att(self) -> int:
        '''
        Returns the dimension of attention
        '''
        if self.hidden_size_att is not None:
            hidden_size_att = self.hidden_size_att
        else:
            hidden_size = self.hidden_size
            assert hidden_size  % 32 == 0, f"hidden_size must be divisible by 32"
            hidden_size_att = hidden_size
        assert hidden_size_att % 32 == 0, f"hidden_size_att must be divisible by 32 ({hidden_size_att})"
        return hidden_size_att
    
    def get_hidden_size_ffn(self) -> int:
        '''
        Returns the dimension of feed forward network
        '''
        if self.hidden_size_ffn is not None:
            hidden_size_ffn = self.hidden_size_ffn
        else:
            hidden_size = self.hidden_size
            assert hidden_size  % 32 == 0, f"hidden_size must be divisible by 32"
            hidden_size_ffn = hidden_size  * 4

        assert hidden_size_ffn % 32 == 0, f"hidden_size_ffn must be divisible by 32"
        return hidden_size_ffn
    
    # def get_n_head(self) -> int:
    #     '''
    #     Returns the number of heads
    #     '''
    #     if self.n_head is not None:
    #         n_head = self.n_head
    #     else:
    #         hidden_size_att = self.get_hidden_size_att()
    #         n_head = self.get_hidden_size_att() // self.head_size
    #         assert hidden_size_att % n_head == 0 ,  f"hidden_size_att must be divisible by head_size ({self.head_size})"
    #
    #     return n_head

    # ---
    # Duplicator & Normalizer
    # ---

    def new_block_config_map(self, **kwargs) -> 'RWKV7BlockConfigMap':
        '''
        Returns a new config map with updated values
        '''

        new_dict = {}
        for key in RWKV7BlockConfigMap.__dataclass_fields__:
            if key in self.__dict__:
                new_dict[key] = self.__dict__[key]
        new_dict.update(kwargs)

        return RWKV7BlockConfigMap(**new_dict)

    @staticmethod
    def normalize(config_map: any) -> 'RWKV7BlockConfigMap':
        '''
        Converts either maps, objs or RWKV7BlockConfigMap
        '''
        if isinstance(config_map, RWKV7BlockConfigMap):
            return config_map
        
        dict_obj = None
        if isinstance(config_map, dict):
            dict_obj = config_map
        elif hasattr(config_map, '__dict__'):
            dict_obj = config_map.__dict__
        
        if dict_obj is not None:
            # Filter for only valeus in RWKV7BlockConfigMap
            new_dict = {}
            for key, value in dict_obj.items():
                if key in RWKV7BlockConfigMap.__dataclass_fields__:
                    new_dict[key] = value
            return RWKV7BlockConfigMap(**new_dict)

        raise ValueError(f"Unsupported config_map type: {type(config_map)}")
