import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import VQADataset
import cnn
import lstm  # TODO


class VQA(nn.Module):
    def __init__(self):
        super(VQA, self).__init__()
        self.cnn = cnn.Xception()
        self.lstm = lstm.LSTM()

    def forward(self, images_batch, questions_batch):
        images_representation = self.cnn(images_batch)
        questions_representation = self.lstm(questions_batch)
        return images_representation, questions_representation


if __name__ == '__main__':
    vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                   questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                                   images_path='data/images',
                                   phase='train')
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

    model = VQA()

    for i_batch, batch in enumerate(train_dataloader):
        """processing for a batch"""

        # stack the images in the batch only to form a [batchsize X 3 X img_size X img_size] tensor
        images_batch_ = torch.stack([sample['image'] for sample in batch], dim=0)
        questions_batch_ = None  # TODO

        batch_image_output = model(images_batch_, questions_batch_)

        breakpoint()
        break
