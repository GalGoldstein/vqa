from vqa_model import evaluate
from dataset import VQADataset
from torch.utils.data import DataLoader
from compute_softscore import compute_targets
import torch
import torch.nn as nn
import os


def evaluate_hw2():
    """
    download data to current directory, convert to .pt files, upload images to RAM and evaluate on validation set
    linux only
    """
    # compute_targets(dir='datashare')

    # download validation set and convert to .pt files instead of .jpg
    # os.system('wget "http://images.cocodataset.org/zips/val2014.zip"')
    # os.system('unzip val2014.zip')
    # os.system('rm val2014.zip')

    # argument create_imgs_tensors will convert all .jpg files to .pt files permanently
    vqa_val_dataset = VQADataset(target_pickle_path='data/cache/val_target.pkl',  # from compute_targets()
                                 questions_json_path='/datashare/v2_OpenEnded_mscoco_val2014_questions.json',
                                 images_path=os.getcwd(),  # current working directory
                                 phase='val', create_imgs_tensors=False, read_from_tensor_files=True, force_mem=True)
    val_dataloader = DataLoader(vqa_val_dataset, batch_size=128, shuffle=False, drop_last=False)

    weights_path = os.path.join(os.getcwd(), 'vqa_id=wpladoyy_epoch_14_val_acc=0.50118.pkl')
    model = torch.load(weights_path)
    vqa_val_dataset.all_questions_to_word_idxs(model)
    vqa_val_dataset.num_classes = model.num_classes

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    val_mean_loss, _, val_mean_acc = evaluate(val_dataloader, model, criterion, 0, vqa_val_dataset)

    return val_mean_loss, val_mean_acc


if __name__ == '__main__':
    loss, accuracy = evaluate_hw2()
    print(f'VALIDATION RESULTS: loss={loss}, accuracy={accuracy}')
