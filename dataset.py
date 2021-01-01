import torch
import os
import json
import pickle
import platform
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import time
import torchvision.transforms.functional as TF


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
            create_imgs_tensors: whether to replace the jpg files with tensors (pt files)
            read_from_tensor_files: whether the images are already tensors (pt files)
            force_mem: upload all images tensors to RAM. can be True only if read_from_tensor_files is True
        """
        self.target = pickle.load(open(target_pickle_path, "rb"))
        self.questions = json.load(open(questions_json_path))['questions']  # [{question obj 1},{question obj 2}..]
        self.original_length = len(self.target)  # the original len before filtering - used for accuracy computations

        # filter questions without answer
        questions_without_answers_idxs = [i for i, answer in enumerate(self.target) if not answer['labels']]
        for index in sorted(questions_without_answers_idxs, reverse=True):
            del self.target[index]
            del self.questions[index]

        self.img_path = images_path
        self.phase = phase
        self.read_pt = read_from_tensor_files
        self.load_imgs_to_mem = force_mem

        if create_imgs_tensors:  # one time creation of img tensors resized
            self.imgs_ids = [int(s[-16:-4]) for s in os.listdir(os.path.join(self.img_path, f'{self.phase}2014'))]
            self.save_imgs_tensors()  # create .pt files and delete .jpg files

        running_on_linux = 'Linux' in platform.platform()
        if not running_on_linux:  # this lines come to make sure we have all needed images in paths (on windows)
            # [-15:-3] for .pt files [-16:-4] for .jpg files
            lower = -15 if read_from_tensor_files else -16
            upper = -3 if read_from_tensor_files else -4
            images = [int(s[lower:upper]) for s in os.listdir(os.path.join(self.img_path, f'{self.phase}2014'))]
            self.target = [target for target in self.target if target['image_id'] in images]
            self.questions = [question for question in self.questions if question['image_id'] in images]

        if force_mem and read_from_tensor_files:
            self.images_tensors = dict()
            self.read_images_to_ram()

    def all_questions_to_word_idxs(self, vqa_model):
        """
        convert a question in natural language to tensor of with words-index
        vqa_model is here because we need the vocabulary
        """
        for question in self.questions:  # preprocess each question
            # question['question'] is natural language question
            question['question'] = vqa_model.gru.words_to_idx(
                vqa_model.gru.preprocess_question_string(question['question']))

    def read_images_to_ram(self):
        """ can be used only if images already converted to tensors"""
        print(f'reading {self.phase} images to RAM')
        for image_id in set([q['image_id'] for q in self.questions]):
            # full path to image
            # the image path contains 12 chars for image id
            path = os.path.join(self.img_path, f'{self.phase}2014',
                                f'COCO_{self.phase}2014_{str(image_id).zfill(12)}.pt')
            self.images_tensors[int(image_id)] = torch.load(path)
            if len(self.images_tensors) % 5000 == 0:
                print(f'{self.phase}, len(self.images_tensors) = {len(self.images_tensors)}')

    def save_imgs_tensors(self):
        """
        take jps original image and resize it, normalize it, and convert it to tensor in float16 (pt file)
        """
        resize = transforms.Resize(size=(224, 224))
        for img_id in self.imgs_ids:
            # the image .jpg path contains 12 chars for image id
            image_id = str(img_id).zfill(12)
            image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.jpg')
            image = Image.open(image_path).convert('RGB')

            # Resize
            image = resize(image)

            # this also divides by 255
            image = TF.to_tensor(image)
            save_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.pt')
            torch.save(image.to(dtype=torch.float16), save_path)  # original type was float32
            os.remove(image_path)  # delete .jpg file

    def load_img_from_path(self, image_path):
        if self.read_pt:  # tensors are already after resize, so we just want to load
            image = torch.load(image_path)

        else:  # load images as jpg files
            image = Image.open(image_path).convert('RGB')

            # Resize
            resize = transforms.Resize(size=(224, 224))
            image = resize(image)

            # this also divides by 255
            image = TF.to_tensor(image)

        return image

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        """
            Return a tuple of 3:
            (image as tensor, question string, answer_dict)

            References:
                1. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
                2. https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
        """
        answer_dict = self.target[idx]
        # e.g. answer_dict =
        # {'question_id': 262148002, 'question_type': 'what is', 'image_id': 262148, 'label_counts': {79: 3, 11: 1},
        #  'labels': [79, 11], 'scores': [0.9, 0.3]}

        # convert answer to soft-score target tensor
        n_classes = self.num_classes
        target = torch.zeros(n_classes)
        for label, score in zip(answer_dict['labels'], answer_dict['scores']):
            target[label] = score

        question_dict = self.questions[idx]
        # e.g. question_dict = {'image_id': 262148, 'question': [0, 4, 6, 8] << tensor, 'question_id': 262148000}
        indexed_question = question_dict['question']
        # e.g. indexed_question = torch.tensor([0, 4, 6, 8, 0, 4, 6, 8, 0, 4, 6, 8, 3, 6]) << length 14 always

        # the image .jpg path contains 12 chars for image id
        image_id = str(question_dict['image_id']).zfill(12)
        extension = 'pt' if self.read_pt else 'jpg'
        image_path = os.path.join(self.img_path, f'{self.phase}2014', f'COCO_{self.phase}2014_{image_id}.{extension}')

        if self.load_imgs_to_mem:  # the images already as tensors on the RAM
            image_tensor = self.images_tensors[int(image_id)]

        else:  # we will load the jpg file and convert it to tensor TODO Yotam to look at
            for i in range(10):
                try:  # full path to image
                    image_tensor = self.load_img_from_path(image_path)
                    break

                except Exception as e:  # TODO Yotam to look at
                    print(f'{e}\n')
                    print(f'Failed in __getitem__ ... trying to load again\n'
                          f'image path: {image_path}\n')
                    time.sleep(3)

        return {'image': image_tensor, 'question': indexed_question, 'answer': target}
