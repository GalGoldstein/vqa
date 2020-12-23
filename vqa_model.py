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
import gru
import pickle
import platform
import time

if 'Linux' in platform.platform():
    import resource

    torch.cuda.empty_cache()
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))  # TODO increase if any problems


# from: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
# torch.multiprocessing.set_sharing_strategy('file_system') # TODO maybe delete?

class VQA(nn.Module):
    def __init__(self, gru_params: dict, label2ans_path: str, target_type: str, img_feature_dim: int):
        super(VQA, self).__init__()
        running_on_linux = 'Linux' in platform.platform()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu' if (torch.cuda.is_available() and not running_on_linux) else self.device

        self.cnn = cnn.CNN().to(self.device)

        self.gru = gru.GRU(gru_params['word_embd_dim'], gru_params['question_hidden_dim'], gru_params['n_layers'],
                           gru_params['train_question_path']).to(self.device)

        self.lbl2ans = pickle.load(open(label2ans_path, "rb"))
        self.num_classes = len(self.lbl2ans)
        self.target_type = target_type

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # gated tanh activation before attention
        self.linear_inside_tanh_attention = nn.Linear(img_feature_dim + gru_params['question_hidden_dim'], 512)
        self.linear_inside_sigmoid_attention = nn.Linear(img_feature_dim + gru_params['question_hidden_dim'], 512)
        # linear layer after gated tanh activation before attention
        self.linear_after_gated_tanh_attention = nn.Linear(512, 1, bias=False)

        # gated tanh activation hidden representation of question
        self.linear_inside_tanh_question = nn.Linear(gru_params['question_hidden_dim'], 512)
        self.linear_inside_sigmoid_question = nn.Linear(gru_params['question_hidden_dim'], 512)

        # gated tanh activation hidden representation of image
        self.linear_inside_tanh_image = nn.Linear(img_feature_dim, 512)
        self.linear_inside_sigmoid_image = nn.Linear(img_feature_dim, 512)

        # gated tanh last
        self.linear_inside_tanh_last = nn.Linear(512, 512)
        self.linear_inside_sigmoid_last = nn.Linear(512, 512)

        # last linear fully connected
        self.fc = nn.Linear(512, self.num_classes, bias=False)

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

    def answers_to_softscore(self, answers_batch, n_classes):
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

        return idx_questions_without_answers, torch.stack(targets, dim=0).to(self.device)

    def forward(self, images_batch, questions_batch):
        # images_representation shape [batch , k , d] where k = number features of image, d = dim of every feature
        images_representation = self.cnn(images_batch)
        questions_last_hidden = [self.gru(self.gru.words_to_idx(question)) for question in questions_batch]
        questions_representation = torch.stack(questions_last_hidden, dim=0).to(self.device)

        expand_dim = [images_representation.shape[1],
                      questions_representation.shape[0],
                      questions_representation.shape[1]]
        concat = torch.cat((images_representation, questions_representation.expand(expand_dim).permute(1, 0, 2)), dim=2)
        gated_tanh_attention = torch.mul(self.tanh(self.linear_inside_tanh_attention(concat)),
                                         self.sigmoid(self.linear_inside_sigmoid_attention(concat)))

        img_features_weights = self.softmax(self.linear_after_gated_tanh_attention(gated_tanh_attention))

        attention_img_features = torch.mul(img_features_weights, images_representation)
        img_sum_weighted_features = torch.sum(attention_img_features, dim=1)

        gated_tanh_img = torch.mul(self.tanh(self.linear_inside_tanh_image(img_sum_weighted_features)),
                                   self.sigmoid(self.linear_inside_sigmoid_image(img_sum_weighted_features)))

        gated_tanh_questions = torch.mul(self.tanh(self.linear_inside_tanh_question(questions_representation)),
                                         self.sigmoid(self.linear_inside_sigmoid_question(questions_representation)))

        pointwise_mul = torch.mul(gated_tanh_img, gated_tanh_questions)

        gated_tanh_mul_product = torch.mul(self.tanh(self.linear_inside_tanh_last(pointwise_mul)),
                                           self.sigmoid(self.linear_inside_sigmoid_last(pointwise_mul)))

        return self.fc(gated_tanh_mul_product)


def evaluate(dataloader, model, criterion, last_epoch_loss, dataset):
    print(f"============ Evaluating on {'validation' if dataset.phase == 'val' else 'train'} set ============")
    model.eval()
    with torch.no_grad():
        accuracy = 0
        epoch_losses = list()
        for i_batch, batch in enumerate(dataloader):
            if model.target_type == 'onehot':
                # answers
                answers_labels_batch = [sample['answer']['label_counts'] for sample in batch]
                target = model.answers_to_one_hot(answers_labels_batch).to(model.device)

                # don't learn from questions without answers
                idx_questions_without_answers = torch.nonzero(target == model.num_classes, as_tuple=False)
                target = target[target != model.num_classes]
            else:  # target_type='softscore'
                answers = [sample['answer'] for sample in batch]
                idx_questions_without_answers, target = model.answers_to_softscore(answers, model.num_classes)

            # stack the images in the batch to form a [batchsize X 3 X img_size X img_size] tensor
            images_batch = torch.stack([sample['image'] for idx, sample in enumerate(batch)
                                        if idx not in idx_questions_without_answers], dim=0).to(model.device)

            # questions
            # Natural language e.g. questions_batch_ = ['How many dogs?'...]
            questions_batch = [sample['question'] for idx, sample in enumerate(batch)
                               if idx not in idx_questions_without_answers]

            output = model(images_batch, questions_batch)

            loss = criterion(output, target)
            epoch_losses.append(float(loss))

            pred = torch.argmax(output, dim=1)
            scores = [{k: v for k, v in zip(sample['answer']['labels'], sample['answer']['scores'])}
                      for idx, sample in enumerate(batch) if idx not in idx_questions_without_answers]

            for i, prediction in enumerate(pred):
                sample_score = scores[i]
                if int(prediction) in sample_score:
                    accuracy += sample_score[int(prediction)]

        acc = accuracy / len(dataset)
        print(f"{'Validation' if dataset.phase == 'val' else 'Train'} accuracy = {round(acc, 5)}")
        cur_epoch_loss = float(np.mean(epoch_losses))
        print(f"{'Validation' if dataset.phase == 'val' else 'Train'} loss = {round(cur_epoch_loss, 5)}")
        if cur_epoch_loss < last_epoch_loss:
            loss_not_improved = False
        else:
            loss_not_improved = True

        return cur_epoch_loss, loss_not_improved, acc


def main():
    pass


# TODO:
#  1. Choose a more simple CNN ??  [V]
#   https://medium.com/swlh/deep-learning-for-image-classification-creating-cnn-from-scratch-using-pytorch-d9eeb7039c12
#  2. BCEWithLogitsLoss and soft_scores_target()  [V]
#  3. Improve data read process (for speed) -
#   - Word to index and target - create them in Dataset
#  4. If continuing to fail - try 'ulimit' to fix the num_workers errors
#  5. Architecture:
#   - F.normalize(x, p=2, dim=1) image representations ??
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

    batch_size = 100 if running_on_linux else 96
    num_workers = 12 if running_on_linux else 0
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=lambda x: x, drop_last=False)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=lambda x: x, drop_last=False)

    word_embd_dim = 300
    img_feature_dim = 256
    question_hidden_dim = 512
    GRU_layers = 1
    gru_params_ = {'word_embd_dim': word_embd_dim, 'question_hidden_dim': question_hidden_dim, 'n_layers': GRU_layers,
                   'train_question_path': train_questions_json_path}

    # TODO target_type?
    target_type = 'softscore'  # either 'onehot' for SingleLabel or 'sofscore' for MultiLabel
    model = VQA(gru_params=gru_params_, label2ans_path=label2ans_path_, target_type=target_type,
                img_feature_dim=img_feature_dim)
    model = model.to(model.device)

    # TODO reduction?
    criterion = nn.CrossEntropyLoss() if model.target_type == 'onehot' else nn.BCEWithLogitsLoss()
    # initial_lr = None
    patience = 4  # how many epochs without val loss improvement to stop training
    optimizer = optim.Adam(model.parameters())  # , lr=initial_lr)  # TODO weight_decay? optimizer?

    print('============ Starting training ============')
    # n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(model.parameters())])
    # print(f'============ # Parameters: {n_params}============')

    print(f'batch_size = {batch_size}\n'
          f'Device: {model.device}\n'
          f'word_embd_dim = {word_embd_dim}\n'
          f'question_hidden_dim = {question_hidden_dim}\n'
          f'GRU_layers = {GRU_layers}\n'
          f'patience = {patience}\n'
          f'target_type = {target_type}\n'
          f'num_workers = {num_workers}\n'
          f'Image model = {model.cnn._get_name()}\n'
          f'Question model = {model.gru._get_name()}\n'
          f'optimizer = {optimizer.__str__()}\n')

    last_epoch_loss = np.inf
    epochs = 100
    count_no_improvement = 0
    for epoch in range(epochs):
        train_epoch_losses = list()
        epoch_start_time = time.time()
        timer_questions = time.time()
        model.train()
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
                idx_questions_without_answers, target = model.answers_to_softscore(answers, model.num_classes)

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

            # TODO we probably have vanishing gradients:
            for name, param in model.named_parameters():
                print(name, param.grad.norm())

            # TODO if exploding gradients:
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

            train_epoch_losses.append(float(loss))
            optimizer.step()

            if i_batch and i_batch % int(1000 / batch_size) == 0:
                print(
                    f'processed {int(1000 / batch_size) * batch_size} questions in {int(time.time() - timer_questions)} '
                    f'secs.  {i_batch * batch_size} / {len(vqa_train_dataset)} total')
                timer_questions = time.time()

        print(f"epoch {epoch + 1}/{epochs} mean train loss: {round(float(np.mean(train_epoch_losses)), 4)}")
        print(f"epoch took {round((time.time() - epoch_start_time) / 60, 2)} minutes")

        cur_epoch_loss, val_loss_didnt_improve, val_acc = \
            evaluate(train_dataloader, model, criterion, last_epoch_loss, vqa_train_dataset)
        # TODO uncomment delete one above
        # evaluate(val_dataloader, model, criterion, last_epoch_loss, vqa_val_dataset)

        # TODO uncomment:
        # if val_loss_didnt_improve:
        #     count_no_improvement += 1
        #     print(f'epoch {epoch + 1} didnt improve val loss. epochs without improvement = {count_no_improvement}')
        # else:
        #     count_no_improvement = 0
        #
        # print(f"============ Saving epoch {epoch + 1} model with validation accuracy = {round(val_acc, 5)} ==========")
        # torch.save(model, os.path.join("weights", f"vqa_model_epoch_{epoch + 1}_val_acc={round(val_acc, 5)}.pth"))
        #
        # last_epoch_loss = cur_epoch_loss
        # if count_no_improvement >= patience:
        #     print(f"========================== Earlystopping epoch {epoch + 1} ==========================")
        #     break
