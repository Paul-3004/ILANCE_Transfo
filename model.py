from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.modules.transformer import _get_clones


class Embedder(nn.Module):
    def __init__(self, nlayers,d_input ,d_model,d_hid,act_func = nn.ReLU(), dtype = torch.float32):
        super(Embedder, self).__init__()
        assert d_hid >  d_model, "Embedder: dimension of hidden layers should be greater than output dimension"
        if nlayers == 1:
            sequence_module = OrderedDict([("input_layer", nn.Linear(d_input,d_model, dtype = dtype))])
            sequence_module.update([("hidden_actfun1", act_func)])
        elif nlayers > 1:
            sequence_module = OrderedDict([("input_layer", nn.Linear(d_input,d_hid, dtype = dtype))])
            sequence_module.update([("hidden_actfun1", act_func)])
            if nlayers > 2:
                for i in range(2, nlayers ):
                    linear = nn.Linear(d_hid,d_hid, dtype =  dtype) #otherwise shares same parameters
                    sequence_module.update([("hidden_linear%d"%i, linear)])
                    sequence_module.update([("hidden_actfun%d"%i, act_func)])
            sequence_module.update([("hidden_linear%d"%nlayers, nn.Linear(d_hid, d_model, dtype = dtype))])

        self.model = nn.Sequential(sequence_module)

    def forward(self,src):
        return self.model(src)

'''
Creates the custom decoder layer for V2 of implementation. 
Input is first processed by nproj projection matrices. Each of those output vector then undergoes 
classical decoder S-A and MHA. The results are then concatenated and fed to the Feed Forward Network
Calling the original nn.TransformerDecoderLayer.__init__() will create sequentially:
    - 1 Self-Attention module 
    - 1 Multi-Head Attention module 
    - 1 2 layered Feed Forward network 
This cusotm decoder layer adds:
    - nproj Projection matrices for the input
    - nproj - 1 additional Self-Attention modules
    - nproj - 1 additional MH Attention modules

    Args:
        d_model: dimension of input vector (after embedding)
        nproj: number of projection matrices to be created
        nhead: number of heads for the attention process
        dim_ff: dimension of the feed forward network
'''
# class CustomDecoderLayer(nn.TransformerDecoderLayer):
#     def __init__(d_model, nproj, nhead, dim_ff, device, dtype):
#         super(CustomDecoderLayer,self).__init__(d_model = d_model, nhead=nhead,dim_feedforward=dim_ff,device = device, dtype = dtype)


class CustomDecoder(nn.Module):
    def __init__(self,layer, nlayers):
        super(CustomDecoder, self).__init__()
        self.nlayers = nlayers
        self.layers = _get_clones(layer, nlayers)
    
    def forward(self,tgt, memory,tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask = tgt_mask,
                                          tgt_key_padding_mask = tgt_key_padding_mask,
                                          memory_key_padding_mask =memory_key_padding_mask, 
                                          tgt_is_causal = tgt_is_causal)
        return output

class CustomDecoderLayer(nn.Module):
    def __init__(self,d_model, d_proj,nproj, nhead, dim_ff_sub, dim_ff_main, batch_first, dtype, device,activation = nn.ReLU(), dropout = 0.1):
        super(CustomDecoderLayer, self).__init__()
        self.dtype = dtype
        self.device = device
        
        dim_proj = d_model // nproj
        assert nproj * dim_proj == d_model, "CustomDecoderLayer: d_model must be divisible by nproj"
        self.nproj = nproj
        self.dim_proj = dim_proj
        
        projection = nn.Linear(d_model, d_proj,dtype = dtype, device = device, bias = False)
        decoder_sublayer = nn.TransformerDecoderLayer(d_proj,nhead,dim_ff_sub,batch_first=batch_first, device = device, dtype = dtype)

        self.projections = _get_clones(projection, nproj)
        self.decoder_layers = _get_clones(decoder_sublayer,nproj)

        #main feed forward
        self.linear1 = nn.Linear(d_model,dim_ff_main)
        self.dropout_ff1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff_main,d_model)
        self.dropout_ff2 = nn.Dropout(dropout)
        #Layer Normalization
        self.norm1 = nn.LayerNorm(d_model,device= device, dtype = dtype)
        self.norm2 = nn.LayerNorm(d_model,device= device, dtype = dtype)



    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal):
        assert self.dim_proj == memory.shape[-1], "CustomDecoderLayer: Dimension of decoder input after projection and encoder output must match."
        size_proj = tgt.shape
        size_proj[-1] = self.dim_proj
        output = torch.zeros(self.nproj, *size_proj, dtype =  self.dtype, device = self.device)
        for i in range(self.nproj):
            proj = self.projections[i](tgt)
            output[i] = self.decoder_layers[i](proj, memory, tgt_mask = tgt_mask,
                                                 tgt_key_padding_mask = tgt_key_padding_mask,
                                                 memory_key_padding_mask = memory_key_padding_mask,
                                                 tgt_is_causal = tgt_is_causal)
        output = torch.concatenate(output.tensor_split(self.nproj, dim = 0), dim = -1).squeeze(0)
        output = self.norm1(output + tgt) # layer normalization
        output = self.norm2(output + self.main_ff_block(output))
        return output

    def main_ff_block(self, x):
        x = self.linear2(self.dropout_ff1(self.linear1(x)))
        return self.dropout_ff2(x)



# Implements a neural network composed of an Embedder converting d_input hits to d_model vectors, 
#           then feeding it as input to transformer and produces the output 
# parameters:
#       Embedder : 
#                  - nlayers_emder: total numbers of layers in the Embedder FFNN
#                  - d_input: numbers of initial features
#                  - act_func_emb: activation function for Embedder FFNN, (default = nn.ReLU)
#                  - d_model: number of features after going through Embedder FFNN, same as features of input encoder
#       Transformer: 
#                  - d_model: shared with Embedder 
#                  - for the rest, see docs at https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html            
class ClustersFinder(nn.Module):
    def __init__(self, dmodel, nhead, d_ff_trsf, device, nlayers_encoder,
                                                                nlayers_decoder,
                                                                custom_decoder = None,
                                                                nlayers_embedder_src: int = 2, 
                                                                d_ff_embedder_src: int = 256,
                                                                d_out_embedder_src: int = 128,
                                                                nlayers_embedder_tgt: int = 2,
                                                                d_ff_embedder_tgt: int = 512,
                                                                d_out_embedder_tgt: int = 256,
                                                                d_input_encoder: int = 6,
                                                                d_input_decoder: int = 8,
                                                                nparticles_max: int = 10, 
                                                                ncharges_max: int = 3,
                                                                DOF_continous: int = 3,
                                                                dtype = torch.float32
                                                                ):
        
        super(ClustersFinder,self).__init__()
        self.device = device
        self.dtype = dtype
        self.input_embedder = Embedder(nlayers = nlayers_embedder_src, d_input=d_input_encoder,d_model=dmodel,d_hid = d_ff_embedder_src, dtype = dtype)
        self.tgt_embedder = Embedder(nlayers = nlayers_embedder_tgt, d_input=d_input_decoder,d_model=dmodel, d_hid = d_ff_embedder_tgt, dtype = dtype)
        self.transformer = nn.Transformer(d_model=dmodel, 
                                          nhead = nhead, 
                                          dim_feedforward= d_ff_trsf,
                                          num_encoder_layers=nlayers_encoder, 
                                          num_decoder_layers=nlayers_decoder, 
                                          custom_decoder= custom_decoder,
                                          batch_first= True,
                                          dtype = dtype,
                                          device = device)

        self.lastlin_charge = nn.Linear(dmodel,ncharges_max, dtype= dtype)
        self.lastlin_pdg = nn.Linear(dmodel,nparticles_max, dtype = dtype)
        self.lastlin_cont = nn.Linear(dmodel,DOF_continous, dtype= dtype)

    '''forward will be called when the __call__ function of nn.Module will be called., used for training
        args:
            src: source sequences, 
                 shape: (N,S,E), N:number of batches, S: sequence length of src, E: embedding dimension
            tgt: target sequences
                 shape: (N,T,E) N, E same as for src, T: seq, length of tgt
            src_padding_mask: mask to avoid attention computed on padding tokens of the source
            tgt_padding_mask: mask to avoid attention computed on padding tokens of the target
            memory_padding_mask: mask to avoid attention computed on padding tokens of memory (output of decoder)
        Note:
            1. if tgt_is_causal is True, tgt_mask is generated automatically as a causal mask'''
    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_padding_mask):
        src = self.input_embedder(src)
        tgt = self.tgt_embedder(tgt)
        output = self.transformer(src = src,tgt = tgt, 
                                  src_key_padding_mask = src_padding_mask, 
                                  tgt_key_padding_mask = tgt_padding_mask,
                                  memory_key_padding_mask = memory_padding_mask,
                                  tgt_mask = self.generate_causal_mask(tgt.shape[1], device = self.device),
                                  tgt_is_causal = True, #generates causal mask for tgt  
                                  )
        
        return (self.lastlin_charge(output), #unnormalised probabilities for charge
                self.lastlin_pdg(output), #unnormalised probabilities for pdg
                self.lastlin_cont(output) ) #Continuous regression

    def	generate_causal_mask(self,sz, device):
        return torch.triu(torch.ones(sz,sz), diagonal = 1).type(torch.bool).to(device = device)
    
    '''Used during inference. Input: source batch,
                              returns: memory: output of the transformer's encoder 
        args: 
            src: source batch, shape: (N,S,E)
            src_key_padding_mask: mask to mask padding tokens in the the batch,
                                  Useful if src input is given as batches during inference.
    '''
    def encode(self, src, src_key_padding_mask):
        return self.transformer.encoder(self.input_embedder(src),src_key_padding_mask = src_key_padding_mask)
    
    '''Used during inference. Input: memory and target,
                              returns: output of the transformer's decoder 
        args: 
            tgt: target batch obtained during inference, shape: (N,T,E)
            memory_key_padding_mask: to mask padding tokens in the batch
                                     Useful if src input is given as batches during inference.
        Note:
            1. Since used during inference, tgt tokens are generated one by one for each batch
               thus, there is no need for a tgt_padding_mask
            2. tgt_mask doesn't need to be supplied if tgt_is_causal is True. Pytorch generates it auto
    '''
    def decode(self, tgt, memory, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer.decoder(self.tgt_embedder(tgt), memory, 
                                        tgt_mask = self.generate_causal_mask(tgt.shape[1],device = self.device),
                                        tgt_is_causal = True, #generates tgt_mask causal 
                                        tgt_key_padding_mask = tgt_key_padding_mask,
                                        memory_key_padding_mask = memory_key_padding_mask)
