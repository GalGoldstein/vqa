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
        # self.device = 'cpu' if (torch.cuda.is_available() and not running_on_linux) else self.device

        self.train_question_path = train_question_path

        # Build questions word dict with number of appearances (counts) of each word
        self.word_dict = self.get_vocabs_counts()
        # init word embeddings. if word has less than min_freq appearances, it will get <unk> embedding.
        # <unk> will serve each test sample word that wasn't seen in the train set.
        # <pad> will serve for padding words, since we pad each question tobe in length 14 (=max length of all qs)
        vocab = Vocab(Counter(self.word_dict), vectors=None, min_freq=2, specials=['<unk>', '<pad>'])
        # set rand vectors and get the weights (the vector embeddings themselves).
        # it means: create len(vocab.stoi) embeddings, each in dimension of word_embd_dim
        words_embeddings_tensor = nn.Embedding(len(vocab.stoi), word_embd_dim).weight.data
        vocab.set_vectors(stoi=vocab.stoi, vectors=words_embeddings_tensor, dim=word_embd_dim)
        self.word_idx_mappings, self.idx_word_mappings, word_vectors = vocab.stoi, vocab.itos, vocab.vectors

        # T.A. pay attention this is not glove or W2V - we initialize the vectors randomly
        # we use from_pretrained function but give the randomly initialized word_vectors
        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=False)

        # define the network as GRU.
        self.encoder = nn.GRU(input_size=word_embd_dim, hidden_size=hidden_dim, num_layers=n_layers,
                              batch_first=True)

    @staticmethod
    def preprocess_question_string(question):
        """
            1. lower all except first word first letter
            2. changing all numbers to letters
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
        """
        return tensor of indexes, with length=number of question words. include <unk> index if a ord is unrecognized
        """
        question = sentence.split(' ')
        question_word_idx_tensor = torch.tensor([self.word_idx_mappings[word] if word in self.word_idx_mappings else
                                                 self.word_idx_mappings['<unk>'] for word in question])
        return question_word_idx_tensor.to(self.device)

    def forward(self, word_idx_tensor):
        trimmed = word_idx_tensor[:14]  # this is not really cut the tensor since no question is longer than 14
        padding_size = 14 - len(trimmed)
        # add the <pad> index * padding_size in the end of the question. i.e. the ad is only in the end of the q.
        padded = torch.cat([trimmed, torch.tensor([self.word_idx_mappings['<pad>']] * padding_size).to(self.device)])
        word_embeddings = self.word_embedding(padded.long())  # nn.Embedding expects to long type
        output, _ = self.encoder(word_embeddings[None, ...])  # supporting only single question and not batch
        return output[0][-1].to(self.device)  # return only last hidden state, of the last layer of GRU


if __name__ == "__main__":
    gru = GRU(100, 1024, 2, 'data/v2_OpenEnded_mscoco_train2014_questions.json')
    out = gru(gru.words_to_idx(' '.join(gru.preprocess_question_string('Where is he looking?'))))

    n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(gru.parameters())])
    print(f'============ # GRU Parameters: {n_params}============')