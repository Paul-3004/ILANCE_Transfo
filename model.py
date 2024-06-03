from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.functional import softmax

class Embedder(nn.Module):
    def __init__(self, nlayers,d_input ,d_model, act_func = nn.ReLU()):
        super().__init__()
        linear = nn.Linear(d_input,d_model, dtype = torch.float64)
        sequence_module = OrderedDict([("input_layer", linear)])
        sequence_module.update([("hidden_actfun1", act_func)])
        for i in range(1, nlayers):
            linear = nn.Linear(d_model,d_model, dtype =  torch.float64) #otherwise shares same parameters
            sequence_module.update([("hidden_actfun%d"%i, act_func)])
            sequence_module.update([("hidden_linear%d"%i, linear)])
        
        self.model = nn.Sequential(sequence_module)

    def forward(self,src):
        return self.model(src)


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
    def __init__(self, dmodel, nhead, nhid_ff_trsf, nlayers_encoder, 
                                                                nlayers_decoder,nlayers_embder, 
                                                                d_input_encoder: int = 6,
                                                                d_input_decoder: int = 8,
                                                                nparticles_max: int = 10, 
                                                                ncharges_max: int = 3,
                                                                DOF_continous: int = 3):
        
        super(ClustersFinder,self).__init__()
        self.input_embedder = Embedder(nlayers = nlayers_embder, d_input=d_input_encoder,d_model=dmodel)
        self.tgt_embedder = Embedder(nlayers = nlayers_embder, d_input=d_input_decoder,d_model=dmodel)
        self.transformer = nn.Transformer(d_model=dmodel, 
                                          nhead = nhead, 
                                          dim_feedforward= nhid_ff_trsf,
                                          num_encoder_layers=nlayers_encoder, 
                                          num_decoder_layers=nlayers_decoder, 
                                          batch_first= True,
                                          dtype = torch.float64)

        self.lastlin_charge = nn.Linear(dmodel,ncharges_max, dtype= torch.float64)
        self.lastlin_pdg = nn.Linear(dmodel,nparticles_max, dtype = torch.float64)
        self.lastlin_cont = nn.Linear(dmodel,DOF_continous, dtype= torch.float64)

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
                                  tgt_is_causal = True, #generates causal mask for tgt  
                                  )  #batch_first to have input shape (N,S,E) or (N,T,E)
        
        return (self.lastlin_charge(output), #unnormalised probabilities for charge
                self.lastlin_pdg(output), #unnormalised probabilities for pdg
                self.lastlin_cont(output) ) #Continuous regression
    
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
                                        tgt_is_causal = True, #generates tgt_mask causal 
                                        tgt_key_padding_mask = tgt_key_padding_mask,
                                        memory_key_padding_mask = memory_key_padding_mask)
