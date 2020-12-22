import math
import torch
import sys
import os
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

if 'Linux' in platform.platform():
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


# from: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
# torch.multiprocessing.set_sharing_strategy('file_system') # TODO maybe delete?


class VQA(nn.Module):
    def __init__(self, lstm_params, label2ans_path, fc_size, target_type):
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
        self.target_type = target_type
        self.activation = nn.ReLU()
        self.fc = nn.Linear(fc_size, self.num_classes)

    def answers_to_one_hot(self, answers_labels_batch):
        """
            answers_labels_batch = [{label:count #people chose this label as answer} ... ]
        """
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

        return self.fc(self.activation(pointwise_mul))


def soft_scores_target(answers_batch, n_classes):
    idx_questions_without_answers = list()
    targets = []
    soft_scores = [{k: v for k, v in zip(sample['labels'], sample['scores'])} for sample in answers_batch]
    for i, soft_score_dict in enumerate(soft_scores):
        if soft_score_dict:
            target = torch.zeros(n_classes)
            for label, score in soft_score_dict.items():
                target[label] = score
            targets.append(target)
        else:
            idx_questions_without_answers.append(i)

    return idx_questions_without_answers, torch.stack(targets, dim=0)


def evaluate(dataLoader, model, criterion, last_epoch_loss, vqa_val_dataset):
    print('============ Evaluating on validation set ============')
    with torch.no_grad():
        accuracy = 0
        val_epoch_losses = list()
        for i_batch_ev, batch_ev in enumerate(dataLoader):
            if model.target_type == 'onehot':
                # answers
                answers_labels_batch__ev = [sample['answer']['label_counts'] for sample in batch_ev]
                target_ev = model.answers_to_one_hot(answers_labels_batch__ev).to(model.device)

                # don't learn from questions without answers
                idx_questions_without_answers_ev = torch.nonzero(target_ev == model.num_classes, as_tuple=False)
                target_ev = target_ev[target_ev != model.num_classes]
            else:  # target_type='softscore'
                answers_ev = [sample['answer'] for sample in batch_ev]
                idx_questions_without_answers_ev, target_ev = soft_scores_target(answers_ev, model.num_classes)

            # stack the images in the batch to form a [batchsize X 3 X img_size X img_size] tensor
            images_batch__ev = torch.stack([sample['image'] for idx, sample in enumerate(batch_ev)
                                            if idx not in idx_questions_without_answers_ev], dim=0).to(model.device)

            # questions
            # Natural language e.g. questions_batch_ = ['How many dogs?'...]
            questions_batch__ev = [sample['question'] for idx, sample in enumerate(batch_ev)
                                   if idx not in idx_questions_without_answers_ev]

            output_ev = model(images_batch__ev, questions_batch__ev)

            loss_ev = criterion(output_ev, target_ev)
            val_epoch_losses.append(float(loss_ev))

            pred = torch.argmax(output_ev, dim=1)
            scores = [{k: v for k, v in zip(sample['answer']['labels'], sample['answer']['scores'])}
                      for sample in batch_ev]

            for i, prediction in enumerate(pred):
                sample_score = scores[i]
                if int(prediction) in sample_score:
                    accuracy += sample_score[int(prediction)]

        val_acc = accuracy / len(vqa_val_dataset)
        print(f'Validation accuracy = {round(val_acc, 5)}')
        cur_epoch_loss_ev = float(np.mean(val_epoch_losses))
        print(f'Validation loss = {round(cur_epoch_loss_ev, 5)}')
        if cur_epoch_loss_ev < last_epoch_loss:
            loss_not_improved = False
        else:
            loss_not_improved = True

        return cur_epoch_loss_ev, loss_not_improved, val_acc


def main():
    pass


# TODO:
#  1. Choose a more simple CNN ??
#   https://medium.com/swlh/deep-learning-for-image-classification-creating-cnn-from-scratch-using-pytorch-d9eeb7039c12
#  2. BCEWithLogitsLoss and soft_scores_target()
#  3. Improve data read process (for speed) -
#   - Word to index and target - create them in Dataset
#  4. If continuing to fail - try 'ulimit' to fix the num_workers errors
#  5. Architecture:
#   - A) Gated tanh on each of the representations
#        Multiplication
#        Gated tanh on the multiplication
#        Fully connected to #classes dim
#        and then BCELossWithLogits
#   - B) F.normalize(x, p=2, dim=1) image representations
#  6. optimizers:
#    A) torch.optim.Adadelta - no need to adjust lr
#    B) torch.optim.Adamax
#  7. Reduce lr to 0.001 as a first try  [V]
#  8. Increase batch size significantly >> 384 ? [V] could make it with 128, maybe can go bit bigger
# nohup python -u vqa_model.py > 1.out&

if __name__ == '__main__':
    # import cProfile
    #
    # PROFFILE = 'prof.profile'
    # cProfile.run('main()', PROFFILE)
    # import pstats
    #
    # p = pstats.Stats(PROFFILE)
    # p.sort_stats('tottime').print_stats(250)
    # main()
    # compute_targets()  TODO uncomment this

    running_on_linux = 'Linux' in platform.platform()

    if running_on_linux:
        vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                       questions_json_path='/datashare/v2_OpenEnded_mscoco_train2014_questions.json',
                                       images_path='/datashare',
                                       force_read=False,
                                       phase='train')
        vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                     questions_json_path='/datashare/v2_OpenEnded_mscoco_val2014_questions.json',
                                     images_path='/datashare',
                                     force_read=False,
                                     phase='val')

        train_questions_json_path = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
        label2ans_path_ = 'data/cache/train_label2ans.pkl'

    else:
        vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                       questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                                       images_path='data/images',
                                       force_read=False,
                                       phase='train')

        vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                     questions_json_path='data/v2_OpenEnded_mscoco_val2014_questions.json',
                                     images_path='data/images',
                                     force_read=False,
                                     phase='val')
        train_questions_json_path = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
        label2ans_path_ = 'data/cache/train_label2ans.pkl'

    batch_size = 128
    num_workers = 12 if running_on_linux else 0
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=lambda x: x, drop_last=False)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=lambda x: x, drop_last=False)

    word_embd_dim = 100
    lstm_hidden_dim = 1280
    LSTM_layers = 1
    lstm_params_ = {'word_embd_dim': word_embd_dim, 'lstm_hidden_dim': lstm_hidden_dim, 'n_layers': LSTM_layers,
                    'train_question_path': train_questions_json_path}

    fc_size = 1280
    target_type = 'onehot'  # either 'onehot' for SingleLabel or 'sofscore' for MultiLabel
    model = VQA(lstm_params=lstm_params_, label2ans_path=label2ans_path_, fc_size=fc_size, target_type=target_type)
    model = model.to(model.device)

    criterion = nn.CrossEntropyLoss() if model.target_type == 'onehot' else nn.BCEWithLogitsLoss()
    initial_lr = 0.001
    patience = 2  # how many epochs without val loss improvement to stop training
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    print('============ Starting training ============')
    n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(model.parameters())])
    print(f'============ # Parameters: {n_params}============')

    print(f'batch_size = {batch_size}\n'
          f'Device: {model.device}\n'
          f'word_embd_dim = {word_embd_dim}\n'
          f'lstm_hidden_dim = {lstm_hidden_dim}\n'
          f'LSTM_layers = {LSTM_layers}\n'
          f'VQA fc_size = {fc_size}\n'
          f'initial_lr = {initial_lr}\n'
          f'patience = {patience}\n'
          f'target_type = {target_type}\n'
          f'num_workers = {num_workers}\n'
          f'Image model = {model.cnn._get_name()}\n'
          f'Question model = {model.lstm._get_name()}\n'
          f'Activation = {model.activation._get_name()}\n'
          f'optimizer = {optimizer.__str__()}\n')

    last_epoch_loss = np.inf
    epochs = 10
    count_no_improvement = 0
    for epoch in range(epochs):
        train_epoch_losses = list()
        epoch_start_time = time.time()
        timer_images = time.time()
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            if model.target_type == 'onehot':
                # answers
                answers_labels_batch_ = [sample['answer']['label_counts'] for sample in batch]
                target = model.answers_to_one_hot(answers_labels_batch_).to(model.device)

                # don't learn from questions without answers
                idx_questions_without_answers = torch.nonzero(target == model.num_classes, as_tuple=False)
                target = target[target != model.num_classes]
            else:  # target_type='softscore'
                answers = [sample['answer'] for sample in batch]
                idx_questions_without_answers, target = soft_scores_target(answers, model.num_classes)

            # stack the images in the batch to form a [batchsize X 3 X img_size X img_size] tensor
            images_batch_ = torch.stack([sample['image'] for idx, sample in enumerate(batch)
                                         if idx not in idx_questions_without_answers], dim=0).to(model.device)

            # questions
            # Natural language e.g. questions_batch_ = ['How many dogs?'...]
            questions_batch_ = [sample['question'] for idx, sample in enumerate(batch)
                                if idx not in idx_questions_without_answers]

            output = model(images_batch_, questions_batch_)
            loss = criterion(output, target)
            loss.backward()
            train_epoch_losses.append(float(loss))
            optimizer.step()

            if i_batch and i_batch % int(1000 / batch_size) == 0:
                print(f'processed {int(1000 / batch_size) * batch_size} questions in {int(time.time() - timer_images)} '
                      f'secs.  {i_batch * batch_size} / {len(vqa_train_dataset)} total')
                timer_images = time.time()

            # TODO this lines evaluate in the middle of the epoch - DELETE
            # if i_batch and i_batch == int(len(train_dataloader) / 2):
            #     # evaluate in the middle of epoch, if no improvement in val loss, reduce lr (lr = lr / 2)
            #     _, reduce_lr, _ = evaluate(val_dataloader, model, criterion, last_epoch_loss, vqa_val_dataset)
            #     if reduce_lr:
            #         print("========================== Reduce Learning Rate ==========================")
            #         print(f"learning rate: {optimizer.param_groups[0]['lr']} > {optimizer.param_groups[0]['lr'] / 2}")
            #         optimizer.param_groups[0]['lr'] /= 2

        print(f"epoch {epoch + 1}/{epochs} mean train loss: {round(float(np.mean(train_epoch_losses)), 4)}")
        print(f"epoch took {round((time.time() - epoch_start_time) / 60, 2)} minutes")

        cur_epoch_loss, val_loss_didnt_improve, val_acc = \
            evaluate(val_dataloader, model, criterion, last_epoch_loss, vqa_val_dataset)

        if val_loss_didnt_improve:
            count_no_improvement += 1
            print(f'epoch {epoch + 1} didnt improve val loss. epochs without improvement = {count_no_improvement}')
        else:
            count_no_improvement = 0

        print(f"============ Saving epoch {epoch + 1} model with validation accuracy = {round(val_acc, 5)} ==========")
        torch.save(model, os.path.join("weights", f"vqa_model_epoch_{epoch + 1}_val_acc={round(val_acc, 5)}.pth"))

        last_epoch_loss = cur_epoch_loss
        if count_no_improvement >= patience:
            print(f"========================== Earlystopping epoch {epoch + 1} ==========================")
            break
