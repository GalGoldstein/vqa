import platform
from vqa_model import evaluate
from dataset import VQADataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn


# TODO GAL: function called "evaluate_hw2()" . The function should load the VQA 2.0 validation set, load
#  your trained network (you can assume that the model file is located in the script folder) and
#  return the average accuracy on the val-set. This function should be written in a separate script.
#  Use this line to load your model:
#  model.load_state_dict(torch.load('model.pkl',map_location=lambda storage, loc: storage))
if __name__ == '__main__':
    running_on_linux = 'Linux' in platform.platform()

    if running_on_linux:
        vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                       questions_json_path='/datashare/v2_OpenEnded_mscoco_train2014_questions.json',
                                       images_path='/datashare',
                                       phase='train', create_imgs_tensors=False, read_from_tensor_files=True)
        vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                     questions_json_path='/datashare/v2_OpenEnded_mscoco_val2014_questions.json',
                                     images_path='/datashare',
                                     phase='val', create_imgs_tensors=False, read_from_tensor_files=True)

        train_questions_json_path = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
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

    batch_size = 100 if running_on_linux else 96
    num_workers = 12 if running_on_linux else 0
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=lambda x: x, drop_last=False)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=lambda x: x, drop_last=False)

    weights_path = 'weights/vqa_model_epoch_50_val_acc=0.27048.pth'
    model = torch.load(weights_path)

    criterion = nn.CrossEntropyLoss() if model.target_type == 'onehot' else nn.BCEWithLogitsLoss(reduction='sum')
    # evaluate(val_dataloader, model, criterion, np.inf, vqa_val_dataset)
    evaluate(train_dataloader, model, criterion, np.inf, vqa_train_dataset)
