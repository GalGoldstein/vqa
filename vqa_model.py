import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VQADataset
from compute_softscore import compute_targets
from torch.nn.utils.weight_norm import weight_norm
import torchvision
import numpy as np
import cnn
import gru
import pickle
import platform
import time


class VQA(nn.Module):
    def __init__(self, gru_params: dict, label2ans_path: str, target_type: str, img_feature_dim: int, padding: int,
                 dropout: float, pooling: str, activation: str):
        """
        gru_params:{word_embd_dim, question_hidden_dim, GRU_layers, train_questions_json_path}
        label2ans_path: path to dictionary connecting between answers and their representing indices
        target_type: soft_scores (can be few possible answers with different scores, to the same question) or
                    one-hot (the most frequent answer has score of 1, the other have 0)
        img_feature_dim: depth of output tensor of the cnn. number of different filters.
        padding: padding size for cnn. if image is 224x224x3, then padding=2 -> regions=5x5, padding=5 -> regions=7x7
                    when regions means number of squares in original image grid, to do attention on.
        dropout: probability to dropout on two places: after the element wise product, and before the last fc.
        pooling: pooling method for the image processing in the cnn network. should be max/average
        activation: activation function for the whole parts ofthe network (e.g.: ReLU)
        """
        super(VQA, self).__init__()
        running_on_linux = 'Linux' in platform.platform()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.cnn = cnn.CNN(padding=padding, pooling=pooling).to(self.device)
        self.padding = padding
        self.pooling = pooling
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.gru = gru.GRU(gru_params['word_embd_dim'], gru_params['question_hidden_dim'], gru_params['n_layers'],
                           gru_params['train_question_path']).to(self.device)
        self.word_embd_dim = gru_params['word_embd_dim']
        self.question_hidden_dim = gru_params['question_hidden_dim']
        self.hidden_dim = gru_params['question_hidden_dim']
        self.n_layers = gru_params['n_layers']

        self.lbl2ans = pickle.load(open(label2ans_path, "rb"))
        self.num_classes = len(self.lbl2ans)
        self.target_type = target_type

        self.softmax = nn.Softmax(dim=1)
        self.activation = activation
        self.relu = nn.ReLU() if self.activation == 'relu' else nn.SELU()
        self.dropout = nn.Dropout(dropout) if self.activation == 'relu' else nn.AlphaDropout()
        self.dropout_p = dropout

        # relu activation before attention
        self.linear_inside_relu_attention = weight_norm(nn.Linear(img_feature_dim + gru_params['question_hidden_dim'],
                                                                  self.hidden_dim), dim=None)
        # linear layer after relu activation before attention
        self.linear_after_relu_attention = weight_norm(nn.Linear(self.hidden_dim, 1, bias=False), dim=None)

        # relu activation hidden representation of image
        self.linear_inside_relu_image = weight_norm(nn.Linear(img_feature_dim, self.hidden_dim), dim=None)

        # relu activation hidden representation of question
        self.linear_inside_relu_question = weight_norm(nn.Linear(gru_params['question_hidden_dim'], self.hidden_dim),
                                                       dim=None)

        # relu activation last
        self.linear_inside_relu_last = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim), dim=None)

        # last linear fully connected
        self.fc = weight_norm(nn.Linear(self.hidden_dim, self.num_classes, bias=False), dim=None)

    def answers_to_one_hot(self, answers_labels_batch):
        """
            answers_labels_batch = [{label:count #people chose this label as answer} ... ]
        """
        all_answers = list()
        for labels_count_dict in answers_labels_batch:
            if labels_count_dict:  # not empty dict
                target_class = max(labels_count_dict, key=labels_count_dict.get)  # only the most common answer gets '1'
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
            else:  # none of the 10 participants answers to that question were passed the min occurrence
                idx_questions_without_answers.append(i)

        return idx_questions_without_answers, torch.stack(targets, dim=0).to(self.device)

    def forward(self, images_batch, questions_batch):
        # images_representation shape [batch_size , k , d] where k = number regions of image, d = dim of every feature
        images_representation = self.cnn(images_batch)
        questions_last_hidden = [self.gru(self.gru.words_to_idx(question)) for question in questions_batch]
        questions_representation = torch.stack(questions_last_hidden, dim=0).to(self.device)

        expand_dim = [images_representation.shape[1],  # k
                      questions_representation.shape[0],  # batch_size
                      questions_representation.shape[1]]  # hidden of question = 512
        concat = torch.cat((images_representation, questions_representation.expand(expand_dim).permute(1, 0, 2)),
                           dim=2)  # [batch_size,k,768]
        relu_attention = self.relu(self.linear_inside_relu_attention(concat))  # [batch_size,k,512]

        img_features_weights = self.softmax(self.linear_after_relu_attention(relu_attention))  # [batch_size,k,1]

        attention_img_features = torch.mul(img_features_weights, images_representation)  # [batch_size,k,d]
        img_sum_weighted_features = torch.sum(attention_img_features, dim=1)  # [batch_size,d]

        relu_imgs = self.relu(self.linear_inside_relu_image(img_sum_weighted_features))  # [batch_size,512]

        relu_questions = self.relu(self.linear_inside_relu_question(questions_representation))  # [batch_size,512]

        pointwise_mul = torch.mul(relu_imgs, relu_questions)  # [batch_size,512]

        relu_mul_product = self.relu(self.linear_inside_relu_last(self.dropout(pointwise_mul)))  # [batch_size,512]

        return self.fc(self.dropout(relu_mul_product))  # [batch_size,n_classes]


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
                sample_score = scores[i]  # {label: score} dict
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


# TODO OPTIMIZATIONS:
#  1. tricks:
#   - Add weight normalization on all nn.Linear() layers (bottom_up git)
#   - Add dropout (look at bottom_up git)
#   - F.normalize(x, p=2, dim=1) image representations ??
#   - question hidden dim 512 >> 1024 (and all the linear layers in VQA)
#   - padding CNN to get bigger dim (current is 256)
#  2. optimizers:
#    A) torch.optim.Adadelta - no need to adjust lr
#    B) torch.optim.Adamax
#  3. More:
#   - learning rate
#   - batch size as big as possible
#  4. Future:
#   - Attention the question (how?)
#  5. Improve data read process (for speed) -
#   - Word to index and target - create them in Dataset
#  Decisions:
#  dropout: {0.0, 0.1, 0.2)}
#  pooling: {Max, Avg}
#  padding: {0, 2}
#  hidden: {512, 1024}  (this number is both the hidden GRU dim and decides on the # of neurons)
#  optimizer: {Adamax, Adadelta}
#  .................................
#  padding=0 or padding=2 (4 first blocks) (5*5 or 7*7) VVVVVVVVVVVVVVV
#  hidden=512 or 1024, VVVVVVVVVVVVVVV
#  Adamax / Adadelta VVVVVVVVVVVVVVV
#  Augmentations (horizontal flip) **Yes** or No XXXXXXXXXX
#  Weight normalization **Yes** or No, XXXXXXXXXX
# nohup python -u vqa_model.py > 1.out&


def main(question_hidden_dim=512, padding=0, dropout_p=0.0, pooling='max', optimizer_name='Adamax', batch_size=64,
         num_workers=0, activation='relu'):
    # compute_targets(dir='datashare')  # TODO uncomment
    # comment next 3 if doesn't want to use wandb
    global vqa_train_dataset
    global vqa_val_dataset
    global use_wandb
    try:
        running_on_linux = 'Linux' in platform.platform()

        if running_on_linux:
            train_questions_json_path = '/home/student/HW2/v2_OpenEnded_mscoco_train2014_questions.json'
            val_questions_json_path = '/home/student/HW2/v2_OpenEnded_mscoco_val2014_questions.json'
            label2ans_path_ = 'data/cache/train_label2ans.pkl'

        else:
            vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                           questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                                           images_path='data/images',
                                           phase='train', create_imgs_tensors=False, read_from_tensor_files=True)

            vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                         questions_json_path='data/v2_OpenEnded_mscoco_val2014_questions.json',
                                         images_path='data/images',
                                         phase='val', create_imgs_tensors=False, read_from_tensor_files=True)
            train_questions_json_path = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
            val_questions_json_path = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
            label2ans_path_ = 'data/cache/train_label2ans.pkl'

        word_embd_dim = 300
        img_feature_dim = 256
        GRU_layers = 1

        # ....................................................................
        question_hidden_dim = question_hidden_dim  # also control the # of neurons in model
        padding = padding  # makes 5*5=25 regions with padding=0 or 7*7=49 regions with padding=2
        dropout_p = dropout_p
        pooling = pooling  # 'max' or 'avg'
        activation = activation
        # ....................................................................
        run_id = ''
        lr = 2e-3
        if use_wandb:
            run = wandb.init()
            run_id = '_id=' + str(run.id)
            print("config:", dict(run.config))
            question_hidden_dim = run.config.hidden  # also control the # of neurons in model
            padding = run.config.padding
            dropout_p = run.config.dropout
            pooling = run.config.pooling  # 'max' or 'avg'
            lr = run.config.lr
            activation = run.config.activation  # 'relu' or 'selu'
            batch_size = run.config.batchsize
        # ....................................................................

        batch_size = batch_size if running_on_linux else 96
        num_workers = num_workers if running_on_linux else 0
        train_dataloader = DataLoader(vqa_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=lambda x: x, drop_last=False)
        val_dataloader = DataLoader(vqa_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=lambda x: x, drop_last=False)

        gru_params_ = {'word_embd_dim': word_embd_dim, 'question_hidden_dim': question_hidden_dim,
                       'n_layers': GRU_layers, 'train_question_path': train_questions_json_path}

        target_type = 'softscore'  # either 'onehot' for SingleLabel or 'sofscore' for MultiLabel
        model = VQA(gru_params=gru_params_, label2ans_path=label2ans_path_, target_type=target_type,
                    img_feature_dim=img_feature_dim, padding=padding, dropout=dropout_p, pooling=pooling,
                    activation=activation)
        model = model.to(model.device)

        criterion = nn.CrossEntropyLoss() if model.target_type == 'onehot' else nn.BCEWithLogitsLoss(reduction='sum')
        # initial_lr = None
        patience = 7  # how many epochs without val loss improvement to stop training
        optimizer = optim.Adamax(model.parameters(), lr=lr) if optimizer_name == 'Adamax' else optim.Adadelta(
            model.parameters())

        print('============ Starting training ============')
        n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(model.parameters())])
        print(f'============ # Parameters: {n_params}============')

        print(f'batch_size = {batch_size}\n'
              f'Device: {model.device}\n'
              f'word_embd_dim = {model.word_embd_dim}\n'
              f'question_hidden_dim, linear layers dim = {model.question_hidden_dim}\n'
              f'GRU_layers = {model.n_layers}\n'
              f'patience = {patience}\n'
              f'pooling = {model.pooling}\n'
              f'padding = {model.padding}\n'
              f'activation = {activation}\n'
              f'dropout probability = {model.dropout_p}\n'
              f'target_type = {model.target_type}\n'
              f'num_workers = {num_workers}\n'
              f'Image model = {model.cnn._get_name()}\n'
              f'Question model = {model.gru._get_name()}\n'
              f'optimizer = {optimizer.__str__()}\n')

        last_epoch_loss = np.inf
        epochs = 4
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
                images_batch_ = torch.stack([model.flip(sample['image'].cuda()) for idx, sample in enumerate(batch)
                                             if idx not in idx_questions_without_answers], dim=0)

                # questions
                # Natural language e.g. questions_batch_ = ['How many dogs?'...]
                questions_batch_ = [sample['question'] for idx, sample in enumerate(batch)
                                    if idx not in idx_questions_without_answers]

                output = model(images_batch_, questions_batch_)
                loss = criterion(output, target)
                loss.backward()

                # if exploding gradients:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25, norm_type=2)

                # printing gradients norms
                # for name, param in model.named_parameters():
                #     print(name, param.grad.norm())

                train_epoch_losses.append(float(loss))
                optimizer.step()

                if not use_wandb and i_batch and i_batch % int(1000 / batch_size) == 0:
                    print(f'processed {int(1000 / batch_size) * batch_size} questions in '
                          f'{int(time.time() - timer_questions)} '
                          f'secs.  {i_batch * batch_size} / {len(vqa_train_dataset)} total')
                    timer_questions = time.time()

            print(f"epoch {epoch + 1}/{epochs} mean train loss: {round(float(np.mean(train_epoch_losses)), 4)}")
            print(f"epoch took {round((time.time() - epoch_start_time) / 60, 2)} minutes")

            torch.cuda.empty_cache()
            cur_epoch_loss, val_loss_didnt_improve, val_acc = \
                evaluate(val_dataloader, model, criterion, last_epoch_loss, vqa_val_dataset)

            if use_wandb:
                wandb.log({"Val Accuracy": val_acc, "Val Loss": cur_epoch_loss, "epoch": epoch + 1})

            # TODO uncomment for the last configuration !
            # train_cur_epoch_loss, _, train_acc = \
            #     evaluate(train_dataloader, model, criterion, last_epoch_loss, vqa_train_dataset)
            # if use_wandb:
            #     wandb.log({"Train Accuracy": train_acc, "Train Loss": train_cur_epoch_loss, "epoch": epoch + 1})

            if val_loss_didnt_improve:
                count_no_improvement += 1
                print(f'epoch {epoch + 1} didnt improve val loss. epochs without improvement = {count_no_improvement}')
            else:
                count_no_improvement = 0

            print(f"========== Saving epoch {epoch + 1} model with validation accuracy = {round(val_acc, 5)} ========")
            torch.save(model, os.path.join("weights", f"vqa{run_id}_epoch_{epoch + 1}_val_acc={round(val_acc, 5)}.pth"))
            torch.cuda.empty_cache()

            last_epoch_loss = cur_epoch_loss
            if count_no_improvement >= patience:
                print(f"========================== Earlystopping epoch {epoch + 1} ==========================")
                break
    except Exception as e:
        print(e)
        print(f'ERROR FAILED')


if __name__ == '__main__':
    while os.system("ps -o cmd= {}".format(126145)) != 256:  # TODO delete
        print('waiting..')
        time.sleep(60)

    time.sleep(60)

    if 'Linux' in platform.platform():
        torch.cuda.empty_cache()
        vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                       questions_json_path='/home/student/HW2/v2_OpenEnded_mscoco_train2014_questions.json',
                                       images_path='/home/student/HW2',
                                       phase='train', create_imgs_tensors=False, read_from_tensor_files=True,
                                       force_mem=True)
        vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                     questions_json_path='/home/student/HW2/v2_OpenEnded_mscoco_val2014_questions.json',
                                     images_path='/home/student/HW2',
                                     phase='val', create_imgs_tensors=False, read_from_tensor_files=True,
                                     force_mem=True)

    if len(sys.argv) > 1 and sys.argv[1] == 'wandb':  # run this code with "python vqa_model.py wandb"
        use_wandb = True
        import logging
        import wandb

        logging.propagate = False
        logging.getLogger().setLevel(logging.ERROR)
        torch.manual_seed(42)  # pytorch random seed
        torch.backends.cudnn.deterministic = True

        # define the hyperparameters
        sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'Val Accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'dropout': {
                    'values': [0.0, 0.1]
                },
                'hidden': {
                    'values': [512, 768, 1024, 1280]
                },
                'padding': {
                    'values': [2]  # 2 >> 5x5 || 5 >> 7x7 (with pic 3x224x224)
                },
                'pooling': {
                    'values': ['max', 'avg']
                },
                'lr': {
                    'distribution': 'uniform',
                    'min': 0.002,
                    'max': 0.01
                },
                'activation': {
                    'values': ['relu']
                },
                'batchsize': {
                    'values': [128]
                }
            }
        }

        # create new sweep
        sweep_id = wandb.sweep(sweep_config, entity="yotammartin", project="vqa")

        # run the agent to execute the code
        wandb.agent(sweep_id, function=main)

    else:  # run this code with "python vqa_model.py"
        use_wandb = True
        import logging
        import wandb

        logging.propagate = False
        logging.getLogger().setLevel(logging.ERROR)
        torch.manual_seed(42)  # pytorch random seed
        torch.backends.cudnn.deterministic = True

        # TODO put here the chosen configuration
        sweep_config = {
            'method': 'grid',
            'metric': {'name': 'Val Accuracy', 'goal': 'maximize'},
            'parameters': {'dropout': {'values': [None]},
                           'hidden': {'values': [None]},
                           'padding': {'values': [None]},
                           'pooling': {'values': [None]},
                           'lr': {'values': [None]},
                           'activation': {'values': [None]}}}

        # create new sweep
        sweep_id = wandb.sweep(sweep_config, entity="yotammartin", project="vqa")

        # run the agent to execute the code
        wandb.agent(sweep_id, function=main)
