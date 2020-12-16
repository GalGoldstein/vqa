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

def get_vocabs_counts(list_of_paths):
    """
        creates dictionary with number of appearances (counts) of each word
    """
    word_dict = defaultdict(int)

    for file_path in list_of_paths:  # paths for json files
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


class QuestionsDataset(Dataset):
    def __init__(self, word_dict=None, word_embd_dim=None, min_freq=1):
        """
        :param path: path to train / test file
        :param word_dict: defaultdict(<class 'int'>, {'Pierre': 1, 'Vinken': 2, ',': 6268,...}
        :param word_embd_dim: dimension of word embedding
        """
        # super().__init__()
        # create Vocab variable just for the ease of using the special tokens and the other nice features
        # like it will create the word_idx_mapping by itself
        vocab = Vocab(Counter(word_dict), vectors=None, min_freq=min_freq)

        # set rand vectors and get the weights (the vector embeddings themselves)
        words_embeddings_tensor = nn.Embedding(len(vocab.stoi), word_embd_dim).weight.data
        vocab.set_vectors(stoi=vocab.stoi, vectors=words_embeddings_tensor, dim=word_embd_dim)

        # take all 3 attributes like in the pre-trained part
        self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = \
            vocab.stoi, vocab.itos, vocab.vectors

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, question_len = self.sentences_dataset[index]
        return word_embed_idx, question_len


def main():
    word_embd_dim = 300
    # hidden_dim = 125
    # epochs = 30
    # learning_rate = 0.01  # Adam's default
    # min_freq = 2  # minimum term-frequency to include in vocabulary, use 1 if you wish to use all words
    # path_train = "train_5700_sentences.labeled"
    # path_test = "test_300_sentences.labeled"

    current_machine_date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    print(f"{current_machine_date_time}\n")



    """TRAIN DATA"""
    train_word_dict = get_vocabs_counts(['mini_v2_Questions_Train_mscoco.json'])
    # for k, v in train_word_dict.items():
    #     print(k, v)

    train = QuestionsDataset(word_dict=train_word_dict, word_embd_dim=word_embd_dim)
    print(train.word_vectors.shape)
    #
    # train_dataloader = DataLoader(train, shuffle=True)
    # model = KiperwasserDependencyParser(train, hidden_dim, MLP_inner_dim, BiLSTM_layers, dropout_layers_probability)
    #
    # """TEST DATA"""
    #
    # test = DependencyDataset(path=path_test, word_dict=train_word_dict, pos_dict=train_pos_dict,
    #                          test=[train.word_idx_mappings, train.pos_idx_mappings], comp_mode=False)
    # test_dataloader = DataLoader(test, shuffle=False)
    #
    # """TRAIN THE PARSER ON TRAIN DATA"""
    # train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list = \
    #     train_kiperwasser_parser(model, train_dataloader, test_dataloader, epochs, learning_rate, weight_decay, alpha,
    #                              path_to_save_model)
    #
    # print(f'\ntrain_accuracy_list = {train_accuracy_list}'
    #       f'\ntrain_loss_list = {train_loss_list}'
    #       f'\ntest_accuracy_list = {test_accuracy_list}'
    #       f'\ntest_loss_list = {test_loss_list}')

if __name__ == "__main__":
    main()