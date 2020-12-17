from torchtext.vocab import Vocab
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import re
import json


class LSTM(nn.Module):
    def __init__(self, word_embd_dim, lstm_hidden_dim, n_layers, train_question_path):
        super(LSTM, self).__init__()
        self.train_question_path = train_question_path

        # Build word dict and init word embeddings #
        self.word_dict = self.get_vocabs_counts()

        # TODO hyper parameters: min_freq, specials?
        #  pre-processing of questions words? lower?
        vocab = Vocab(Counter(self.word_dict), vectors=None, min_freq=1, specials=['<unk>'])
        # set rand vectors and get the weights (the vector embeddings themselves)
        words_embeddings_tensor = nn.Embedding(len(vocab.stoi), word_embd_dim).weight.data
        vocab.set_vectors(stoi=vocab.stoi, vectors=words_embeddings_tensor, dim=word_embd_dim)
        self.word_idx_mappings, self.idx_word_mappings, word_vectors = vocab.stoi, vocab.itos, vocab.vectors

        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=False)

        self.encoder = nn.LSTM(input_size=word_embd_dim, hidden_size=lstm_hidden_dim, num_layers=n_layers, batch_first=True)

    def get_vocabs_counts(self):
        """
            creates dictionary with number of appearances (counts) of each word
        """
        word_dict = defaultdict(int)

        with open(self.train_question_path) as json_file:
            data = json.load(json_file)
            for q_object in data['questions']:
                words = re.split(' ', q_object['question'])
                for word in words:
                    if word in word_dict.keys():
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
        return word_dict

    def words_to_idx(self, sentence: str):
        question = sentence.split(' ')
        question_word_idx_tensor = torch.tensor([self.word_idx_mappings[i] if i in self.word_idx_mappings else
                                                 self.word_idx_mappings['<unk>'] for i in question])
        return question_word_idx_tensor

    def forward(self, sentence: str):
        word_idx_tensor = self.words_to_idx(sentence)
        word_embeddings = self.word_embedding(word_idx_tensor)

        output, _ = self.encoder(word_embeddings[None, ...])  # TODO currently supporting only a single sentence
        return output[0][-1]  # return only last hidden state, of the last layer of LSTM


if __name__ == "__main__":
    lstm = LSTM(100, 1024, 2, 'data/v2_OpenEnded_mscoco_train2014_questions.json')
    out = lstm('Where is he looking?')
