import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_toknizers(config, ds, lang):
    toknizer_path = Path(config['toknizer_file'].format(lang))

    if not Path.exists(toknizer_path):
        toknizer = Tokenizer(WordLevel(unk_token = ['UNK']))
        toknizer.pre_toknizer  = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        toknizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        toknizer.save(str(toknizer_path))
    else:
        toknizer = Tokenizer.from_file(str(toknizer_path))
    
    return toknizer

def get_ds(config):
    ds_raw = load_dataset('cfilt/iitb-english-hindi', f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
    
    # Build Toknizers,
    toknizer_src = get_or_build_toknizers(config, ds_raw, config['lang_src'])
    toknizer_tgt = get_or_build_toknizers(config, ds_raw, config['lang_tgt'])

    #keep 90% for training and 10% for validation,
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])



