import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, toknizer_src, toknizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.toknizer_src = toknizer_src
        self.toknizer_tgt = toknizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([toknizer_src.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos_token = torch.Tensor([toknizer_tgt.token_to_id(['[EOD]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([toknizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.toknizer_src.encode(src_text).ids
        dec_input_tokens = self.toknizer_tgt.decode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 :
            raise ValueError("sentence is too long.")
        
        # Add SOS and EOS to the text,
        encoder_input  = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype= torch.int64)
            ]
        )

        # Add SOS decoder input,
        decoder_input  = torch.cat(
            [
                self.eos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (what we expect as ouput from the decoder),
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensror([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
         
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # seq_len
            "decoder_input":decoder_input, # seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1_seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int)
    return mask == 0