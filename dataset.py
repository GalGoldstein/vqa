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
import torch.nn.functional as F
import random
import time
import torchvision.transforms.functional as TF
import gru
import sys


class VQADataset(Dataset):
    """Visual Question Answering v2 dataset."""

    def __init__(self, target_pickle_path: str, questions_json_path: str, images_path: str, phase: str,
                 create_imgs_tensors: bool = False, read_from_tensor_files: bool = True, force_mem: bool = False):
        """
        parameters:
            target_pickle_path: (str) path to pickle file produced with compute_softscore.py
                e.g. 'val_target.pkl'
            train_questions_json_path: (str) path to json with questions
                e.g. 'v2_OpenEnded_mscoco_val2014_questions.json'
            images_path: (str) path to dir containing 'train2014' and 'val2014' folders
            phase: (str) 'train' / 'val'
        """
        self.target = pickle.load(open(target_pickle_path, "rb"))
        self.questions = json.load(open(questions_json_path))['questions']
        for question in self.questions:
            question['question'] = ' '.join(gru.GRU.preprocess_question_string(question['question']))
        self.img_path = images_path
        self.phase = phase
        self.read_pt = read_from_tensor_files
        self.load_imgs_to_mem = force_mem

        if create_imgs_tensors:  # one time creation of img tensors resized
            self.imgs_ids = [int(s[-16:-4]) for s in os.listdir(os.path.join(self.img_path, f'{self.phase}2014'))]
            self.save_imgs_tensors()

        running_on_linux = 'Linux' in platform.platform()
        if not running_on_linux:  # this 3 lines come to make sure we have all needed images in paths
            images = [int(s[-15:-3]) for s in os.listdir(os.path.join(self.img_path, f'{self.phase}2014'))]
            self.target = [target for target in self.target if target['image_id'] in images]
            self.questions = [question for question in self.questions if question['image_id'] in images]

        if force_mem:
            self.images_tensors = dict()
            self.read_images()

    def read_images(self):
        image_ids = set([q['image_id'] for q in self.questions])
        resize = transforms.Resize(size=(224, 224))
        for image_id in image_ids:
            # full path to image
            # the image .jpg path contains 12 chars for image id
            image_id = str(image_id).zfill(12)
            image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.pt')
            img = TF.to_pil_image(torch.load(image_path).to(dtype=torch.float32))
            self.images_tensors[int(image_id)] = TF.to_tensor(resize(img)).to(dtype=torch.float16)
            del img

    def save_imgs_tensors(self):
        for img_id in self.imgs_ids:
            # the image .jpg path contains 12 chars for image id
            image_id = str(img_id).zfill(12)
            image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.jpg')
            image = Image.open(image_path).convert('RGB')

            # Resize
            resize = transforms.Resize(size=(299, 299))
            image = resize(image)

            # this also divides by 255
            image = TF.to_tensor(image)
            save_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.pt')
            torch.save(image.to(dtype=torch.float16), save_path)
            os.remove(image_path)  # delete .jpg file

    def load_img_from_path(self, image_path):
        if self.read_pt:
            image = torch.load(image_path)

        else:
            image = Image.open(image_path).convert('RGB')

            # Resize
            resize = transforms.Resize(size=(299, 299))
            image = resize(image)

            # this also divides by 255
            image = TF.to_tensor(image)

        # horizontal flip augmentation  TODO - cannot do this to float16
        # if self.phase == 'train' and random.random() > 0.5:
        #     image = TF.hflip(image)
        return image

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

        # the image .jpg path contains 12 chars for image id
        image_id = str(question_dict['image_id']).zfill(12)
        extension = 'pt' if self.read_pt else 'jpg'
        image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.{extension}')

        if self.load_imgs_to_mem:  # load the image from RAM
            image_tensor = self.images_tensors[int(image_id)]

        else:
            for i in range(10):
                try:  # full path to image
                    image_tensor = self.load_img_from_path(image_path)
                    break

                except Exception as e:
                    print(f'{e}\n')
                    print(f'Failed in __getitem__ ... trying to load again\n'
                          f'image path: {image_path}\n')
                    time.sleep(3)

        return {'image': image_tensor, 'question': question_string, 'answer': answer_dict}


if __name__ == '__main__':
    running_on_linux = 'Linux' in platform.platform()
    if running_on_linux:
        train_questions_json_path = '/home/student/HW2/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = '/home/student/HW2/v2_OpenEnded_mscoco_val2014_questions.json'
        images_path = '/home/student/HW2'
    else:
        train_questions_json_path = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
        val_questions_json_path = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
        images_path = 'data/images'

    num_workers = 12 if running_on_linux else 0

    vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                   questions_json_path=train_questions_json_path,
                                   images_path=images_path,
                                   phase='train', create_imgs_tensors=False, read_from_tensor_files=True,
                                   force_mem=True)
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=16, shuffle=True,
                                  collate_fn=lambda x: x, num_workers=num_workers, drop_last=False)

    vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',
                                 questions_json_path=val_questions_json_path,
                                 images_path=images_path,
                                 phase='val', create_imgs_tensors=False, read_from_tensor_files=True, force_mem=True)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=16, shuffle=False,
                                collate_fn=lambda x: x, num_workers=num_workers, drop_last=False)

    for i_batch, batch in enumerate(train_dataloader):
        print(i_batch, batch)
        print('\n\n\n\n\n\n\n\n\n')
        break

    for i_batch, batch in enumerate(val_dataloader):
        print(i_batch, batch)
        break
