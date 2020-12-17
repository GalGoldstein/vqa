import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VQADataset
from compute_softscore import compute_targets
import numpy as np
import cnn
import lstm
import pickle
import platform


class VQA(nn.Module):
    def __init__(self, lstm_params, label2ans_path, fc_size=2048):
        super(VQA, self).__init__()
        self.cnn = cnn.Xception()
        self.lstm = lstm.LSTM(lstm_params['word_embd_dim'],
                              lstm_params['lstm_hidden_dim'],
                              lstm_params['n_layers'],
                              lstm_params['train_question_path'])

        self.lbl2ans = pickle.load(open(label2ans_path, "rb"))
        self.num_classes = len(self.lbl2ans)
        self.fc = nn.Linear(fc_size, self.num_classes + 1)  # +1 for questions with answer that has no class

    def answers_to_one_hot(self, answers_labels_batch):
        all_answers = list()
        for labels_count_dict in answers_labels_batch:
            if labels_count_dict:  # not empty dict
                target_class = max(labels_count_dict, key=labels_count_dict.get)
            else:  # TODO continue
                target_class = self.num_classes  # last class is used for the questions without an answer
            all_answers.append(target_class)

        return torch.tensor(all_answers)

    def forward(self, images_batch, questions_representation):
        images_representation = self.cnn(images_batch)
        pointwise_mul = torch.mul(images_representation, questions_representation)

        return self.fc(pointwise_mul)


if __name__ == '__main__':
    compute_targets()
    running_on_linux = 'Linux' in platform.platform()

    print()
    print("VQADataset")
    if running_on_linux:
        vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                       questions_json_path='/datashare/v2_OpenEnded_mscoco_train2014_questions.json',
                                       images_path='/datashare',
                                       phase='train')

        vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                     questions_json_path='/datashare/v2_OpenEnded_mscoco_val2014_questions.json',
                                     images_path='/datashare',
                                     phase='val')

        train_questions_json_path = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
        label2ans_path_ = 'data/cache/train_label2ans.pkl'

    else:
        vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                       questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                                       images_path='data/images',
                                       phase='train')

        vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                     questions_json_path='data/v2_OpenEnded_mscoco_val2014_questions.json',
                                     images_path='data/images',
                                     phase='val')

        train_questions_json_path = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
        label2ans_path_ = 'data/cache/train_label2ans.pkl'

    print("DataLoader")
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

    lstm_params_ = {'word_embd_dim': 100, 'lstm_hidden_dim': 2048, 'n_layers': 1,
                    'train_question_path': train_questions_json_path}

    print("model")
    model = VQA(lstm_params=lstm_params_, label2ans_path=label2ans_path_)
    model.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.device = 'cpu' if (torch.cuda.is_available() and not running_on_linux) else model.device  # TODO delete
    model = model.to(model.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    print()
    print(model.device)
    for epoch in range(50):
        epoch_losses = list()
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # stack the images in the batch only to form a [batchsize X 3 X img_size X img_size] tensor
            images_batch_ = torch.stack([sample['image'] for sample in batch], dim=0).to(model.device)

            # questions
            questions_batch_ = [sample['question'] for sample in batch]
            questions_representation_ = torch.stack([model.lstm(question) for question in questions_batch_],
                                                    dim=0).to(model.device)

            # answers
            answers_labels_batch_ = [sample['answer']['label_counts'] for sample in batch]
            target = model.answers_to_one_hot(answers_labels_batch_).to(model.device)

            output = model(images_batch_, questions_representation_)
            loss = criterion(output, target)
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
        print(f"epoch {epoch + 1} mean loss: {round(float(np.mean(epoch_losses)), 4)}")
