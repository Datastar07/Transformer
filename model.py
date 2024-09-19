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

        # Not update this vector as backpropagation process,
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


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h."

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv

        self.w_o = nn.Linear(d_model, d_model) # wv
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key , value, mask, dropout: nn.Dropout) -> None:
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch ,h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return (attention_score @ value), attention_score



    def forward(self,q, k, v, mask):
        query = self.w_q(q)  # (batch ,seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch ,seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch ,seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch ,seq_len, d_model) --> (batch, seq_len , h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key= key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)


        x,self.attention_scores = MultiHeadAttentionBlock.attention(query=query, key=key, value=value,mask=mask, dropout=self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1 , self.h * self.d_k)

        
        return self.w_o(x)




