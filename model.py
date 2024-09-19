import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model: int, voc_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = voc_size
        self.embedding = nn.Embedding(voc_size, d_model)
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape,(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape(seq_len, 1)
        position =  torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # Apply sine term to even postion and cos term to odd postion,
        pe[:,0::2]  = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  #(1,seq_len, d_model)

        # not update this vector as backpropagation process,
        self.register_buffer("pe", pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1],:]).required_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):
        mean = x.mean(dim =-1, keepdim = True)
        std = x.std(dim =-1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

        
