import torch
import torch.nn as nn
from torch.utils.data import DataLoader

"""
https://medium.com/swlh/deep-learning-for-image-classification-creating-cnn-from-scratch-using-pytorch-d9eeb7039c12
"""


class CNN(nn.Module):
    def __init__(self, padding=0, pooling='max'):
        super(CNN, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # formula to calc length (or width) of tensor after the conv layer:
        # (2 * padding_value) + previous_len_of_row_before_conv_layer - (kernel_size - 1) = len_of_row_after_conv_layer
        # for example: 2*2 + 224 - (3 - 1) = 226
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=padding), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=padding), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=padding), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=padding), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            # note 1: We don't want padding in the last layers, since then the final output tensor will include
            # padding pixels, and this tensor is much smaller than the original image (e.g. 5x5x256)
            # note 2: BatchNorm2d is recommended after each conv layer, and the parameter is need to get is the current
            # number of filters (which is depth of tensor, which is output value of the last conv layer
        )

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x = self.convolutions(x)  # x.shape =  [batch_size, 3, 224, 224]
            x = x.permute(0, 2, 3, 1)  # [batch_size, 256, 5, 5] -> [batch_size, 5, 5, 256]
            x = x.reshape([x.size(0), x.size(1) * x.size(2), -1])  # [batch_size, 5, 5, 256]  -> [batch_size, 25, 256]
            return x  # [batch_size, 25, 256]. 25=K=Number of regions, 256=d=Dimension of each region


if __name__ == "__main__":
    from dataset import VQADataset

    vqa_train_dataset = VQADataset(target_pickle_path='data/cache/train_target.pkl',
                                   questions_json_path='data/v2_OpenEnded_mscoco_train2014_questions.json',
                                   images_path='data/images',
                                   phase='train', create_imgs_tensors=False, read_from_tensor_files=True)
    train_dataloader = DataLoader(vqa_train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN(padding=2, pooling='max').to(device)

    n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(cnn.parameters())])
    print(f'============ # CNN parameters: {n_params}============')

    for i_batch, batch in enumerate(train_dataloader):
        """processing for a single image"""
        image = batch[0]['image']
        single_image_output = cnn(image[None, ...].to(device))

        """processing for a batch"""
        # stack the images in the batch only to form a [batchsize X 3 X img_size X img_size] tensor
        images_batch = torch.stack([sample['image'] for sample in batch], dim=0)
        batch_image_output = cnn(images_batch.to(device))
        breakpoint()
        break
