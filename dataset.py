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

        
