from torchtext.vocab import Vocab
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import platform
import re
import json

NUMBERS = {'0': 'zero',
           '1': 'one',
           '2': 'two',
           '3': 'three',
           '4': 'four',
           '5': 'five',
           '6': 'six',
           '7': 'seven',
           '8': 'eight',
           '9': 'nine',
           '10': 'ten'}


class GRU(nn.Module):
    def __init__(self, word_embd_dim, hidden_dim, n_layers, train_question_path):
        super(GRU, self).__init__()

        running_on_linux = 'Linux' in platform.platform()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu' if (torch.cuda.is_available() and not running_on_linux) else self.device

        self.train_question_path = train_question_path

        # Build word dict and init word embeddings #
        self.word_dict = self.get_vocabs_counts()

        # TODO hyper parameters: min_freq, specials? (i set min_freq=2 to train the unk token)
        vocab = Vocab(Counter(self.word_dict), vectors=None, min_freq=2, specials=['<unk>', '<pad>'])
        # set rand vectors and get the weights (the vector embeddings themselves)
        words_embeddings_tensor = nn.Embedding(len(vocab.stoi), word_embd_dim).weight.data
        vocab.set_vectors(stoi=vocab.stoi, vectors=words_embeddings_tensor, dim=word_embd_dim)
        self.word_idx_mappings, self.idx_word_mappings, word_vectors = vocab.stoi, vocab.itos, vocab.vectors

        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=False)

        self.encoder = nn.GRU(input_size=word_embd_dim, hidden_size=hidden_dim, num_layers=n_layers,
                              batch_first=True)

    @staticmethod
    def preprocess_question_string(question):
        """
            1. only numbers and letters
            2. lower all except first word first letter
            3. any word with number higher than 10 >> <number>. any word with number lower to 10: e.g. 9 >>'nine'
        """
        result = question[0].upper() + question[1:].lower()
        # changing 0,1,...,10 to the name of the number e.g. ten >> 10
        words = [NUMBERS[word] if word in NUMBERS else word for word in result.split(' ')]
        # if we still have a number in a word conver it to general number '<number>'
        result = [('<number>' if any(char.isdigit() for char in word) else re.sub(r'[\W_]+', '', word))
                  for word in words]
        return result

    def get_vocabs_counts(self):
        """
            creates dictionary with number of appearances (counts) of each word
        """
        word_dict = defaultdict(int)

        with open(self.train_question_path) as json_file:
            data = json.load(json_file)
            for q_object in data['questions']:
                words = self.preprocess_question_string(q_object['question'])
                for word in words:
                    if word in word_dict.keys():
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
        return word_dict

    def words_to_idx(self, sentence: str):
        question = sentence.split(' ')
        question_word_idx_tensor = torch.tensor([self.word_idx_mappings[word] if word in self.word_idx_mappings else
                                                 self.word_idx_mappings['<unk>'] for word in question])
        return question_word_idx_tensor.to(self.device)

    def forward(self, word_idx_tensor):
        trimmed = word_idx_tensor[:14]
        padding_size = 14 - len(trimmed)
        padded = torch.cat([trimmed, torch.tensor([self.word_idx_mappings['<pad>']] * padding_size).to(self.device)])
        word_embeddings = self.word_embedding(padded.long())
        output, _ = self.encoder(word_embeddings[None, ...])  # currently supporting only a single sentence
        return output[0][-1].to(self.device)  # return only last hidden state, of the last layer of GRU


if __name__ == "__main__":
    gru = GRU(100, 1024, 2, 'data/v2_OpenEnded_mscoco_train2014_questions.json')
    out = gru(gru.words_to_idx(' '.join(gru.preprocess_question_string('Where is he looking?'))))

    n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(gru.parameters())])
    print(f'============ # GRU Parameters: {n_params}============')

    # lstm = LSTM(100, 1024, 2, 'data/v2_OpenEnded_mscoco_train2014_questions.json')
    # out = lstm(lstm.words_to_idx(' '.join(lstm.preprocess_question_string('Where is he looking?'))))

    # n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(lstm.parameters())])
    # print(f'============ # LSTM Parameters: {n_params}============')
