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
import time


class VQA(nn.Module):
    def __init__(self, lstm_params, label2ans_path, fc_size):
        super(VQA, self).__init__()
        running_on_linux = 'Linux' in platform.platform()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu' if (torch.cuda.is_available() and not running_on_linux) else self.device

        # self.cnn = cnn.Xception().to(self.device)
        self.cnn = cnn.MobileNetV2().to(self.device)

        self.lstm = lstm.LSTM(lstm_params['word_embd_dim'],
                              lstm_params['lstm_hidden_dim'],
                              lstm_params['n_layers'],
                              lstm_params['train_question_path']).to(self.device)

        self.lbl2ans = pickle.load(open(label2ans_path, "rb"))
        self.num_classes = len(self.lbl2ans)
        self.fc = nn.Linear(fc_size, self.num_classes + 1)  # +1 for questions with answer that has no class

    def answers_to_one_hot(self, answers_labels_batch):
        all_answers = list()
        for labels_count_dict in answers_labels_batch:
            if labels_count_dict:  # not empty dict
                target_class = max(labels_count_dict, key=labels_count_dict.get)
            else:
                target_class = self.num_classes  # last class is used for the questions without an answer
            all_answers.append(target_class)

        return torch.tensor(all_answers)

    def forward(self, images_batch, questions_batch):
        images_representation = self.cnn(images_batch)
        questions_last_hidden = [self.lstm(self.lstm.words_to_idx(question)) for question in questions_batch]
        questions_representation = torch.stack(questions_last_hidden, dim=0).to(self.device)

        pointwise_mul = torch.mul(images_representation, questions_representation)

        return self.fc(pointwise_mul)


if __name__ == '__main__':
    # compute_targets()

    running_on_linux = 'Linux' in platform.platform()

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

    batch_size = 32
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    lstm_params_ = {'word_embd_dim': 100, 'lstm_hidden_dim': 1280, 'n_layers': 1,
                    'train_question_path': train_questions_json_path}

    model = VQA(lstm_params=lstm_params_, label2ans_path=label2ans_path_, fc_size=1280)
    model = model.to(model.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print('============ Starting training ============')
    n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(model.parameters())])
    print(f'============ # Parameters: {n_params}============')

    print(f'Device: {model.device}')

    epochs = 50
    for epoch in range(epochs):
        epoch_losses = list()
        epoch_start_time = time.time()
        timer_images = time.time()
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # stack the images in the batch to form a [batchsize X 3 X img_size X img_size] tensor
            images_batch_ = torch.stack([sample['image'] for sample in batch], dim=0).to(model.device)

            # questions
            questions_batch_ = [sample['question'] for sample in batch]  # Natural language e.g. 'How many dogs?'

            # answers
            answers_labels_batch_ = [sample['answer']['label_counts'] for sample in batch]
            target = model.answers_to_one_hot(answers_labels_batch_).to(model.device)

            output = model(images_batch_, questions_batch_)
            loss = criterion(output, target)
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()

            if i_batch % int(1000 / batch_size) == 0:
                print(f'processed {int(1000 / batch_size) * batch_size} questions in {int(time.time() - timer_images)} '
                      f'secs.  {i_batch * batch_size} / {len(vqa_train_dataset)} total')
                timer_images = time.time()
        print(f"epoch {epoch + 1}/{epochs} mean loss: {round(float(np.mean(epoch_losses)), 4)}")
        print(f"epoch took {round((time.time() - epoch_start_time) / 60, 2)} minutes")

        with torch.no_grad():
            accuracy = 0
            for i_batch, batch in enumerate(val_dataloader):
                # stack the images in the batch to form a [batchsize X 3 X img_size X img_size] tensor
                images_batch_ = torch.stack([sample['image'] for sample in batch], dim=0).to(model.device)

                # questions
                questions_batch_ = [sample['question'] for sample in batch]  # Natural language e.g. 'How many dogs?'

                # answers
                answers_labels_batch_ = [sample['answer']['label_counts'] for sample in batch]
                target = model.answers_to_one_hot(answers_labels_batch_).to(model.device)

                output = model(images_batch_, questions_batch_)

                pred = torch.argmax(output, dim=1)
                scores = [{k: v for k, v in zip(sample['answer']['labels'], sample['answer']['scores'])}
                          for sample in batch]

                for i, prediction in enumerate(pred):
                    sample_score = scores[i]
                    if int(prediction) in sample_score:
                        accuracy += sample_score[int(prediction)]

            val_acc = accuracy / len(vqa_val_dataset)
            print(f'Validation accuracy = {round(val_acc, 5)}')

# TODO:
#  1. choose a cnn with less params ??
#   https://medium.com/swlh/deep-learning-for-image-classification-creating-cnn-from-scratch-using-pytorch-d9eeb7039c12
#  2. early stopping, reduce lr on plateau
#  3. smaller words vocabulary in LSTM ??
