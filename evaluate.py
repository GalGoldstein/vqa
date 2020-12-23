import platform
from vqa_model import VQA
from vqa_model import evaluate
from dataset import VQADataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn

if __name__ == '__main__':
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

    batch_size = 100 if running_on_linux else 16
    num_workers = 12 if running_on_linux else 0
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=lambda x: x, drop_last=False)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=lambda x: x, drop_last=False)

    weights_path = 'weights/vqa_model_epoch_6_val_acc=0.24086_reduction=mean.pth'
    model = torch.load(weights_path)

    # TODO reduction
    criterion = nn.CrossEntropyLoss() if model.target_type == 'onehot' else nn.BCEWithLogitsLoss(reduction='sum')
    evaluate(val_dataloader, model, criterion, np.inf, vqa_val_dataset)
    evaluate(train_dataloader, model, criterion, np.inf, vqa_train_dataset)
