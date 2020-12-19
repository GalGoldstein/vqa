import torch
import os
import io
import numpy as np
import json
import pickle
import platform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
import lstm
import sys


class VQADataset(Dataset):
    # TODO hyperparameters:
    #  1. Resize image
    #  2. Any kind of augmentation? crop? flip?

    """Visual Question Answering v2 dataset."""

    def __init__(self, target_pickle_path: str, questions_json_path: str, images_path: str, phase: str,
                 force_read: bool):
        """
        parameters:
            target_pickle_path: (str) path to pickle file produced with compute_softscore.py
                e.g. 'val_target.pkl'
            train_questions_json_path: (str) path to json with questions
                e.g. 'v2_OpenEnded_mscoco_val2014_questions.json'
            images_path: (str) path to dir containing 'train2014' and 'val2014' folders
            phase: (str) 'train' / 'val'
            force_read: (bool) if True loads all images to RAM. False reads image only by request (call to __getitem__)
        """
        self.target = pickle.load(open(target_pickle_path, "rb"))
        self.questions = json.load(open(questions_json_path))['questions']
        for question in self.questions:
            question['question'] = ' '.join(lstm.LSTM.preprocess_question_string(question['question']))
        self.img_path = images_path
        self.phase = phase
        self.force_read = force_read

        # TODO delete next 3 lines: only for verifying everything works (verify image in path)
        running_on_linux = 'Linux' in platform.platform()
        if not running_on_linux:
            images = [int(s[15:-4]) for s in os.listdir(os.path.join(self.img_path, f'{self.phase}2014'))]
            self.target = [target for target in self.target if target['image_id'] in images]
            self.questions = [question for question in self.questions if question['image_id'] in images]

        if self.force_read:
            self.images_tensors = dict()
            self.read_images()

    def read_images(self):
        image_ids = list(set([q['image_id'] for q in self.questions]))[:1000]  # TODO
        for image_id in image_ids:
            # full path to image
            # the image .jpg path contains 12 chars for image id
            image_id = str(image_id).zfill(12)
            image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.jpg')
            image = Image.open(image_path).convert('RGB')

            # TODO - set parameter for resize in args
            #  what is the size we want?
            # Resize
            resize = transforms.Resize(size=(224, 224))
            image = resize(image)

            # this also divides by 255 TODO we can normalize too
            self.images_tensors[int(image_id)] = TF.to_tensor(image)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        """
            Return a tuple of 3:
            (image as tensor, question string, answer)

            References:
                1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
                2. https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
        """
        answer_dict = self.target[idx]
        # e.g. answer_dict =
        # {'question_id': 262148002, 'question_type': 'what is', 'image_id': 262148, 'label_counts': {79: 3, 11: 1},
        #  'labels': [79, 11], 'scores': [0.9, 0.3]}

        question_dict = self.questions[idx]
        # e.g. question_dict = {'image_id': 262148, 'question': 'Where is he looking?', 'question_id': 262148000}
        question_string = question_dict['question']
        # e.g. question_string = 'Where is he looking?'

        if self.force_read:  # upload all to RAM
            image_tensor = self.images_tensors[question_dict['image_id']]

        else:
            # the image .jpg path contains 12 chars for image id
            image_id = str(question_dict['image_id']).zfill(12)

            # full path to image
            image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.jpg')

            image = Image.open(image_path).convert('RGB')

            # TODO - set parameter for resize in args
            #  what is the size we want?
            # Resize
            resize = transforms.Resize(size=(224, 224))
            image = resize(image)

            # this also divides by 255 TODO we can normalize too
            image_tensor = TF.to_tensor(image)

        return {'image': image_tensor, 'question': question_string, 'answer': answer_dict}


# TODO can we make use of any of the following functions?
# class MyDataset(Dataset):
#     def __init__(self, image_paths, target_paths, train=True):
#         self.image_paths = image_paths
#         self.target_paths = target_paths
#
#     def transform(self, image, mask):
#         # Resize
#         resize = transforms.Resize(size=(520, 520))
#         image = resize(image)
#         mask = resize(mask)
#
#         # Random crop
#         i, j, h, w = transforms.RandomCrop.get_params(
#             image, output_size=(512, 512))
#         image = TF.crop(image, i, j, h, w)
#         mask = TF.crop(mask, i, j, h, w)
#
#         # Random horizontal flipping
#         if random.random() > 0.5:
#             image = TF.hflip(image)
#             mask = TF.hflip(mask)
#
#         # Random vertical flipping
#         if random.random() > 0.5:
#             image = TF.vflip(image)
#             mask = TF.vflip(mask)
#
#         # Transform to tensor
#         image = TF.to_tensor(image)
#         mask = TF.to_tensor(mask)
#         return image, mask
#
#     def __getitem__(self, index):
#         image = Image.open(self.image_paths[index])
#         mask = Image.open(self.target_paths[index])
#         x, y = self.transform(image, mask)
#         return x, y
#
#     def __len__(self):
#         return len(self.image_paths)


if __name__ == '__main__':
    vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                   questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                                   images_path='data/images',
                                   force_read=True,
                                   phase='train')
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=16, shuffle=True,
                                  collate_fn=lambda x: x)
    print(f'sizeof self.images_tensors 1k images: {sys.getsizeof(vqa_train_dataset.images_tensors)}')
    exit(777)  # TODO
    for i_batch, batch in enumerate(train_dataloader):
        print(i_batch, batch)
        break

    vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                 questions_json_path='data/v2_OpenEnded_mscoco_val2014_questions.json',
                                 images_path='data/images',
                                 force_read=True,
                                 phase='val')
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)
    for i_batch, batch in enumerate(val_dataloader):
        print(i_batch, batch)
        break
