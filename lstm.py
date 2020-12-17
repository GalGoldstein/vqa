from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset
from collections import Counter
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import re
import matplotlib.pyplot as plt
import time
import os
import json
import dataset

UNKNOWN_TOKEN = "<unk_token>"
SPECIAL_TOKENS = [UNKNOWN_TOKEN]

def get_vocabs_counts(list_of_paths):
    """
        creates dictionary with number of appearances (counts) of each word
    """
    word_dict = defaultdict(int)

    for file_path in list_of_paths:  # paths for Questions.json files
        with open(file_path) as json_file:
            data = json.load(json_file)
            for q_object in data['questions']:
                words = re.split(' ', q_object['question'])
                for word in words:
                    if word in word_dict.keys():
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
    return word_dict


def get_vocabs_counts_list_of_words(list_of_words):
    """
        creates dictionary with number of appearances (counts) of each word
    """
    word_dict = defaultdict(int)

    for question in list_of_words:  # paths for Questions.json files
        for word in question:
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    return word_dict


class my_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, BiLSTM_layers):
        super(my_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=BiLSTM_layers, batch_first=True)

    def forward(self, sample):  # this is required function. can't change its name
        lstm_out, _ = self.encoder(sample)
        return lstm_out[0][-1]  # return only last hidden state, of the last layer of LSTM


if __name__ == "__main__":
    word_embd_dim = 100
    hidden_dim = 125
    BiLSTM_layers = 2  # article's default
    dropout_layers_probability = 0.0  # nn.LSTM default

    ### Build word dict and init word embeddings ###
    train_dataloader, val_dataloader = dataset.get_data_loaders()
    train_questions_batch = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        for sample in sample_batched:
            train_questions_batch.append(sample['question'].split(' '))
    train_word_dict = get_vocabs_counts_list_of_words(train_questions_batch)

    vocab = Vocab(Counter(train_word_dict), vectors=None, min_freq=1)
    # set rand vectors and get the weights (the vector embeddings themselves)
    words_embeddings_tensor = nn.Embedding(len(vocab.stoi), word_embd_dim).weight.data
    vocab.set_vectors(stoi=vocab.stoi, vectors=words_embeddings_tensor, dim=word_embd_dim)
    word_idx_mappings, idx_word_mappings, word_vectors = vocab.stoi, vocab.itos, vocab.vectors
    # print(word_vectors.shape)
    print(word_idx_mappings)

    ### new sample into lstm model ###
    lstm = my_LSTM(word_embd_dim, hidden_dim, BiLSTM_layers)
    # one sample
    for i_batch, batch in enumerate(train_dataloader):
        """processing for a single image"""
        question = batch[0]['question'].split(' ')
        question_indexes = [word_idx_mappings[i] if i in word_idx_mappings.keys() else
                            word_idx_mappings['<unk>'] for i in question]
        question_embeddings = torch.stack([word_vectors[i] for i in question_indexes], dim=0)
        questions_output = lstm(question_embeddings[None, ...])

        print(questions_output.shape)
    # one batch
    # all_questions_embeddings = []
    # for i_batch, batch in enumerate(train_dataloader):
    #     """processing for a single image"""
    #     for index_s, s in enumerate(batch):
    #         question = batch[index_s]['question'].split(' ')
    #         question_indexes = [train.word_idx_mappings[i] if i in train.word_idx_mappings.keys() else
    #                     train.word_idx_mappings['<unk>'] for i in question]
    #         question_embeddings = torch.stack([train.word_vectors[i] for i in question_indexes], dim=0)
    #
    #         # print(question_embeddings)
    #         # word_embeddings = torch.FloatTensor(question_embeddings)
    #         # print(word_embeddings)
    #         all_questions_embeddings.append(question_embeddings)
    #     all_questions_embeddings = torch.stack(all_questions_embeddings, dim=0)
    #     print(all_questions_embeddings.shape)

    # questions_output = lstm(all_questions_embeddings)
    # print(questions_output)
