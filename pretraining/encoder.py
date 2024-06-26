import torch
import torch.nn as nn
import numpy as np
from ..model import Embedder
from torch.nn.modules.transformer import _get_clones, TransformerEncoderLayer, TransformerEncoder

class EncoderPretraining(nn.Module):
    def __init__(self, d_model, nhead, n_SA, n_CA, d_ff, nclusters_max, device, dtype = torch.float32):
        super(EncoderPretraining, self).__init__()
        Att_FFN = TransformerEncoderLayer(d_model=d_model, nhead = nhead,dim_feedforward=d_ff,batch_first=True, dtype = dtype)
        self.SA_layers = _get_clones(Att_FFN, n_SA)
        self.CA_layers = _get_clones(Att_FFN, n_CA)
        self.CLS = nn.Parameter(torch.randn(1,1,d_model))

        self.embedder = Embedder(2,d_model,d_model, 2* d_model, dtype=dtype)
        self.lastlin = nn.Linear(d_model, nclusters_max, dtype= dtype)

    def forward(self, src, CA_mask, src_key_padding_mask):
        for SA_layer in self.SA_layers:
            src = SA_layer(src,src_mask = None,src_key_padding_mask = src_key_padding_mask)
        
        CLS_src = torch.concatenate([self.CLS, src], dim = 1)
        CA_key_padding_mask = _get_CA_mask(src_key_padding_mask)

        for CA_layer in self.CA_layers:
            CLS_src = CA_layer(CLS_src, src_mask = CA_mask, src_key_padding_mask = CA_key_padding_mask)
        
        
        
    def _get_CA_mask(self, src_key_padding_mask):
        pass

