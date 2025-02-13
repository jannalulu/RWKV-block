import torch, math
from torch import nn
from torch import Tensor
from typing import Union

from .qwerky7_config_map import Qwerky7ConfigMap
from ..block.qwerky7_layer_block import Qwerky7LayerBlock

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

class Qwerky7Model(nn.Module):
    '''
    Qwerky7 Model architecture
    Simplified implementation

    Note: This EXCLUDES the head layer, keeping in line with the HF format convention
    '''

    def __init__(self, config: Union[Qwerky7ConfigMap, any, None] = None):
        # Initialize the base class
        super().__init__()

        # Normalize the config
        configMap:Qwerky7ConfigMap = Qwerky7ConfigMap.normalize(config)
        self.configMap = configMap

        # Get the required prop
        num_hidden_layers = configMap.num_hidden_layers
        vocab_size = configMap.vocab_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        hidden_size = configMap.hidden_size
        padding_idx = configMap.padding_idx
        head_size = configMap.head_size

        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx).to(device, dtype=dtype)

        # main layers
        self.layers = nn.ModuleList(
            [Qwerky7LayerBlock(config.new_block_config_map(layer_id=layer_idx)) for layer_idx in range(config.num_hidden_layers)]
        )

        # ln_out
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device, dtype=dtype)
        # self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # init state tuning support
        if configMap.init_state_wkv:
            stateTuneList = [None]*num_hidden_layers
            for i in range(num_hidden_layers):
                stateTuneList[i] = nn.ParameterDict({
                    "wkv": nn.Parameter(torch.zeros(hidden_size // head_size, head_size, head_size, device=device, dtype=torch.float)),
                })
            self.init_state = nn.ParameterList(stateTuneList)

    def reset_parameters(self):
        '''
        Reset the parameters of the model, to an initial state used for training a model from scratch
        '''
        # Get the required prop
        configMap = self.configMap
        num_hidden_layers = configMap.num_hidden_layers
        vocab_size = configMap.vocab_size
        device = configMap.get_device(None)
        dtype = configMap.get_dtype('bfloat16')
        hidden_size = configMap.hidden_size
        head_size = configMap.head_size
        
        # Iterate and reset the layers
        for i in range(num_hidden_layers):
            self.layers[i].reset_parameters()

        # Reinit the Embedding layer
        self.embed_tokens.reset_parameters()

        # Reinit the RMSNorm
        self.norm.weight.data.fill_(1.0)

        # if self.lm_head is not None:
        #     self.lm_head.reset_parameters()

        # Reinit the init state tuning support
        if configMap.init_state_wkv:
            if self.init_state is None:
                stateTuneList = [None]*num_hidden_layers
                for i in range(num_hidden_layers):
                    stateTuneList[i] = nn.ParameterDict({
                        "wkv": nn.Parameter(torch.zeros(hidden_size // head_size, head_size, head_size, device=device, dtype=torch.float)),
                    })
                self.init_state = nn.ParameterList(stateTuneList)
            else:
                for i in range(num_hidden_layers):
                    self.init_state[i]["wkv"].data.copy_(torch.zeros(hidden_size // head_size, head_size, head_size, device=device, dtype=torch.float))


    def load_from_model_state_dict(self, state_dict: dict, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        for i, layers in enumerate(self.layers):
            layers.load_from_model_state_dict(state_dict, i, non_blocking=non_blocking)

        self.embed_tokens.weight.data.copy_(state_dict['model.embed_tokens.weight'], non_blocking=non_blocking)
        self.norm.weight.data.copy_(state_dict['model.norm.weight'], non_blocking=non_blocking)
        # self.norm.bias.data.copy_(state_dict['model.norm.bias'], non_blocking=non_blocking)
        
        if self.configMap.init_state_wkv:
            for i in range(self.configMap.num_hidden_layers):
                if 'model.init_state.'+str(i)+'.wkv' in state_dict:
                    self.init_state[i]["wkv"].data.copy_(state_dict['model.init_state.'+str(i)+'.wkv'], non_blocking=True)

    ### ---
    ###
    ### Init state handling
    ###
    ### ---

    def get_init_state(self, batch_size:int=1, skip_init_state:bool=False) -> list[torch.Tensor]:
        '''
        Get an initialized copy of the model state, for the given batch size
        '''
        # Get required configs
        hidden_size = self.configMap.hidden_size
        init_state_wkv = self.configMap.init_state_wkv
        num_hidden_layers = self.configMap.num_hidden_layers
        head_size = self.configMap.head_size

        # Prepare the initial state
        init_state = [ None for i in range(num_hidden_layers) ]
        for i in range(num_hidden_layers):
            device = self.layers[i].self_attn.q_proj.weight.data.device

            # Use the saved init_state if enabled
            # TODO: Consider letting the wkv_state dtype be a parameter
            wkv_state = torch.zeros(batch_size, hidden_size // head_size, head_size, head_size, device=device, dtype=torch.float)
            if init_state_wkv and skip_init_state == False:
                init_wkv = self.init_state[i]["wkv"]
                for b in range(batch_size):
                    wkv_state[b][:] = init_wkv

            # Prepare the state
            init_state[i] = wkv_state
        return init_state

    ### ---
    ###
    ### Model Forward
    ###
    ### ---

    def forward(
        self, idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor] = None,  
        ret_stateList:list[torch.Tensor] = None,
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Forward the layer set, given the input tokens and the last state
        Last state is a list of time mix wkv state

        Returns a pair of the output embedding and the next state
        '''
        # Prepare the state, with the batch size
        if prv_stateList is None:
            prv_stateList = self.get_init_state(idx.shape[0])

        # If no return state is set, let _forward_internal, set it up
        if ret_stateList is None:
            ret_stateList = [ None for i in range(self.configMap.num_hidden_layers) ]
            return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=False)

        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=overwrite_ret_tensor)
    
    def _forward_internal_embeddings(
            self, x_hidden_state:torch.Tensor, 
            prv_stateList:list[torch.Tensor],  
            ret_stateList:list[torch.Tensor],
            overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Internal forward operations, which assumes the state is already initialized.
        And uses the x_hidden_state as the input (bypassing the embedding layer)
        And returns the output embedding before the head layer

        Due to the lack of safety checks, this should not be used directly
        '''
        # Get the batch and input length
        batch_size = x_hidden_state.shape[0]
        x_input_length = x_hidden_state.shape[1]

        # Initialize the v_first
        v_first = None

        # Force overwrite_ret_tensor to False, if ret_stateList is None
        if ret_stateList is None:
            overwrite_ret_tensor = False

        # Get the forward chunk size, and the chunk count
        forward_chunk_size = self.configMap.forward_chunk_size
        forward_chunk_count = math.ceil( x_input_length / forward_chunk_size )

        # Iterate the layers, compute the x_hidden_state
        for i, layer in enumerate(self.layers):
            
            # Single pass, optimized
            if forward_chunk_count <= 1:
                x_hidden_state, last_layer_state, v_first = self._forward_layer_hook(layer, x_hidden_state, prv_stateList[i], v_first)
            else:
                # Sadly, we need to chunk
                new_x_hidden_state_arr = [None]*forward_chunk_count
                v_first_arr = [None]*forward_chunk_count if v_first is None else None
                last_layer_state = prv_stateList[i]

                # Iterate the chunks
                for j in range(forward_chunk_count):
                    start = j * forward_chunk_size
                    endin = min( start + forward_chunk_size, x_input_length )

                    new_x_hidden_state, last_layer_state, v_first_part = self._forward_layer_hook(
                        layer, 
                        x_hidden_state[:,start:endin], 
                        last_layer_state, 
                        v_first[:, start:endin] if v_first is not None else None
                    )

                    # Save the chunk
                    new_x_hidden_state_arr[j] = new_x_hidden_state
                    if v_first_arr is not None:
                        v_first_arr[j] = v_first_part

                # Merge the chunks
                x_hidden_state = torch.cat(new_x_hidden_state_arr, dim=1)
                if v_first_arr is not None:
                    v_first = torch.cat(v_first_arr, dim=1)

            # last_layer_state = prv_stateList[i]
            # Overwrite tensor if needed
            if overwrite_ret_tensor:
                ret_stateList[i][:] = last_layer_state
            else:
                ret_stateList[i] = last_layer_state
                
        # Final layer norm, without the head
        x_hidden_state = x_hidden_state.to(self.norm.weight.device, non_blocking=True)
        x_hidden_state = self.norm(x_hidden_state)

        # Return the output and the state list
        return x_hidden_state, ret_stateList
        
    def _forward_layer_hook(self, 
            layer:Qwerky7LayerBlock, 
            x_hidden_state:torch.Tensor, 
            prv_stateList:list[torch.Tensor], 
            v_first:torch.Tensor
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        Forward layer hook operation, that is easily overridable.
        To implement gradient checkpointing for use in various trainers
        '''
        x_hidden_state = x_hidden_state.to(layer.input_layernorm.weight.device, non_blocking=True)
        return layer(x_hidden_state, prv_stateList, v_first)
    
    def _forward_internal(
        self, idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor],  
        ret_stateList:list[torch.Tensor],
        overwrite_ret_tensor:bool=False
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Internal forward operations, which assumes the state is already initialized
        Due to the lack of safety checks, this should not be used directly
        '''
        # Lets get the embedding
        idx = idx.to(self.embed_tokens.weight.device, non_blocking=True)
        x_hidden_state = self.embed_tokens(idx)

        # Forward the layer layers
        x_output_embedding, retStateList = self._forward_internal_embeddings(x_hidden_state, prv_stateList, ret_stateList, overwrite_ret_tensor)

        # Return the output and the state list
        return x_output_embedding, retStateList

    @torch.compile(mode="default")
    def forward_with_default_compile(
        self, idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor],
        ret_stateList:list[torch.Tensor],
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        # Forward internally
        return self._forward_internal(idx, prv_stateList, ret_stateList, overwrite_ret_tensor=True)
  
    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(
        self, in_idx:torch.Tensor, 
        prv_stateList:list[torch.Tensor]
    ) -> tuple[torch.Tensor,list[torch.Tensor]]:
        '''
        Compiled varient of the forward function, requires previous state to be passed
        '''
        return self._forward_internal(in_idx, prv_stateList, None, overwrite_ret_tensor=False)
