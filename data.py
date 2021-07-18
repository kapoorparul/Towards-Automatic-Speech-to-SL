# coding: utf-8
"""
Data module
"""
import sys
import os
import os.path
from typing import Optional
import io
import numpy as np
# from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field
import torch

from constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import build_vocab, Vocabulary

import librosa

import nltk
from nltk.stem import   WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
wordnet_lemmatizer = WordNetLemmatizer()

# Load the Regression Data
# Data format should be parallel .txt files for src, trg and files
# Each line of the .txt file represents a new sequence, in the same order in each file
# src file should contain a new source input on each line
# trg file should contain skeleton data, with each line a new sequence, each frame following on from the previous
# Joint values were divided by 4 to move to the scale of -1 to 1
# Each joint value should be separated by a space; " "
# Each frame is partioned using the known trg_size length, which includes all joints (In 2D or 3D) and the counter
# Files file should contain the name of each sequence on a new line


def load_data(cfg: dict, mode='train') -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """

    data_cfg = cfg["data"]
    is_train = mode=='train'
    
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    nonreg_trg_lang = data_cfg["nonreg_trg"]
    files_lang = data_cfg.get("files", "files")
    
    # Train, Dev and Test Path
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    level = "char" #"word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    
    # Target size is plus one due to the counter required for the model
    src_size = cfg["model"]["src_size"]
    trg_size = cfg["model"]["trg_size"] + 1
    
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    num_sec=data_cfg.get("num_sec", 6) 

    # Files field is just a raw text field
    files_field = data.RawField()

    def tokenize_features(features):
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    # Source field is a tokenised version of the source words
    src_field = data.Field(sequential=True,
                           use_vocab=False,
                           dtype=torch.float32,
                           batch_first=True,
                           include_lengths=True,
                           pad_token=torch.ones((src_size,))*TARGET_PAD,)
                           

    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((trg_size,))*TARGET_PAD,)
    
    ## For text translation                           
    nonreg_trg_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)
    
    # Create the Training Data, using the SignProdDataset
    train_data = SignProdDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang, "." + nonreg_trg_lang, "." + files_lang),
                                    fields=(src_field, reg_trg_field, nonreg_trg_field, files_field),
                                    trg_size=trg_size,
                                    src_size = src_size,
                                    skip_frames=skip_frames, is_train = is_train,
                                    num_sec = num_sec)
                           

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    nonreg_trg_min_freq = data_cfg.get("nonreg_trg_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = [None]*src_size
    
    # Create a target vocab just as big as the required target vector size -
    trg_vocab = [None]*trg_size
    
    nonreg_trg_vocab = build_vocab(field="nonreg_trg", min_freq=nonreg_trg_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file) 
    
    # Create the Validation Data
    dev_data = SignProdDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang, "." + nonreg_trg_lang, "." + files_lang),
                                  trg_size=trg_size,
                                  src_size = src_size,
                                  fields=(src_field, reg_trg_field, nonreg_trg_field, files_field),
                                  skip_frames=skip_frames, is_train = is_train,
                                  num_sec=num_sec)
    

    # Create the Testing Data
    test_data = SignProdDataset(path=test_path,
                                    exts=("." + src_lang, "." + trg_lang, "." + nonreg_trg_lang, "." + files_lang),
                                    trg_size=trg_size,
                                    src_size = src_size,
                                    fields=(src_field, reg_trg_field, nonreg_trg_field, files_field),
                                    skip_frames=skip_frames, is_train = is_train,
                                    num_sec=num_sec)
    
    src_field.vocab = src_vocab
    nonreg_trg_field.vocab = nonreg_trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab, nonreg_trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch

def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


# Main Dataset Class
class SignProdDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    def __init__(self, path, exts, fields, trg_size, src_size, num_sec, skip_frames=1, is_train=True, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]),('nonreg_trg', fields[2]), ('file_paths', fields[3])]

        src_path, trg_path, nonreg_trg_path, file_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []

        src_fps, trg_fps = 100, 25
                
        # load tar files path
        tar = torch.load(path)
        print("loaded.... ", path)
        
        num_vids=0
        
        for file_paths, (_) in tar.items():
            src = tar[file_paths]["src"]
            trg = tar[file_paths]["trg"]
            nonreg_trg_line = tar[file_paths]["text"]

            if src.shape[0] > src_fps *  num_sec:
                continue

            # lemmantize the text translations
            src_wrds = nonreg_trg_line.split(" ")
            lemma=[]
            for w in src_wrds:
                if(wordnet_lemmatizer.lemmatize(w)=='wa'):
                    lemma.append(w)
                elif(wordnet_lemmatizer.lemmatize(w)=='ha'):
                    lemma.append(w)
                else:
                    lemma.append(wordnet_lemmatizer.lemmatize(w))
            
            nonreg_trg_line = " ".join(lemma) 
                        

            examples.append(data.Example.fromlist(
                [src[:], trg[:num_sec*trg_fps], nonreg_trg_line, file_paths], fields))
            num_vids+=1

            
        print("Num of {} videos is {}".format(path.split('/')[-1], num_vids))

        super(SignProdDataset, self).__init__(examples, fields, **kwargs)
