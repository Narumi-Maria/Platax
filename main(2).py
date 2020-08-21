import numpy as np
import pandas as pd
from PIL import Image
import os
import datetime
import sys
from tqdm import tqdm
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

# COMMON CONFIG
DEVICE = "cuda"
WEIGHT_DIR = r"weight"  # save checkpoint here
TRAIN_FILE_PTH = r"af2020cv-2020-05-09-v5-dev\training.csv"
TEST_FILE_PTH = r"af2020cv-2020-05-09-v5-dev\annotation.csv"
PLATAX_IMAGE_PTH = r"af2020cv-2020-05-09-v5-dev\data"

# ABLATION OPTIONS
INIT_FROM_IMAGENET = True
# download from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
IMAGENET_PRETRAIN_MODEL_PATH = r"E:\GitHub项目\pretrained model\imagenet_resnet18.pth"
PAD_TO_SQUARE = True

# TRAINING CONFIG
CONTINUE = False
LOAD_WEIGHT_PATH = r'weight.pkl'  # if CONTINUE, then load weight from here
BATCHSIZE = 64
LR = 0.01
EPOCH = 25
TEST_FREQ = 1  # test per #epoch
SPECIES = 20
CHECKPOINT_SAVE_FREQ = 1


def get_resnet18():
    net = resnet18()
    if INIT_FROM_IMAGENET:
        net.load_state_dict(torch.load(IMAGENET_PRETRAIN_MODEL_PATH))
        print('\n[load imagenet pretrained model successfully!]\n')
    modified_net = nn.Sequential(*list(net.children())[:-1])
    return modified_net


class ClassifyModel(nn.Module):
    def __init__(self, n_class=20):
        super(ClassifyModel, self).__init__()
        self.backbone = get_resnet18()
        self.extra_layer = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.extra_layer(out)
        return out


def make_data(file_pth):
    train_data = pd.read_csv(open(file_pth), sep=',')
    data_list = []
    train_x_name = []
    train_y = []
    for column, row in train_data.iterrows():
        data_list.append((row['FileID'], row['SpeciesID']))
    random.shuffle(data_list)
    for i, data in enumerate(data_list):
        train_x_name.append(data[0])
        train_y.append(data[1])
    return train_x_name, train_y


class ImageDataset(Dataset):
    def __init__(self, train_x_name, train_y, mode='train'):
        super(ImageDataset, self).__init__()
        print("\n[make %s dataset -- #sample %d]\n" %
              (mode, len(train_x_name)))
        self.image_list = train_x_name
        self.lable_list = train_y
        if mode == 'train':
            self.transforms_ = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms_ = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(PLATAX_IMAGE_PTH,
                                      "%s.jpg" % self.image_list[idx]))
        if PAD_TO_SQUARE:
            h, w = img.size
            value = abs(h - w) // 2
            if h < w:
                pad = (value, 0, value, 0)
            else:
                pad = (0, value, 0, value)
            img = transforms.Pad(pad)(img)

        img = self.transforms_(img)
        label = self.lable_list[idx]
        label = torch.tensor(label, dtype=torch.long)
        return img, label


def get_loader(mode, train_x_name, train_y, shuffle, bs):
    data = ImageDataset(train_x_name, train_y, mode=mode)
    return DataLoader(data, batch_size=bs, shuffle=shuffle)


def train():
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    net = ClassifyModel(SPECIES).to(DEVICE)

    if CONTINUE:
        net.load_state_dict(torch.load(LOAD_WEIGHT_PATH))
        print('\n[continue to train from %s!]\n' % LOAD_WEIGHT_PATH)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    train_x_name, train_y = make_data(TRAIN_FILE_PTH)
    test_x_name, test_y = make_data(TEST_FILE_PTH)

    train_loader = get_loader('train', train_x_name,
                              train_y, shuffle=True, bs=BATCHSIZE)
    test_loader = get_loader('test', test_x_name, test_y,
                             shuffle=False, bs=BATCHSIZE)

    optimizer = optim.SGD(net.parameters(), lr=LR,
                          momentum=0.9, weight_decay=1e-4)

    start_time = datetime.datetime.now()

    for epoch in range(1, 1+EPOCH):
        for n_iter, (img, label) in enumerate(train_loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()

            predict_label = net(img)
            loss = criterion(predict_label, label)

            loss.backward()
            optimizer.step()
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.2f] time: %s " % (
                epoch, EPOCH, n_iter, len(train_loader), loss.item(), datetime.datetime.now() - start_time), end='')

        if epoch % TEST_FREQ == 0:
            net.eval()
            print()
            with torch.no_grad():
                correct = 0
                total = 0
                for data, label in tqdm(test_loader):
                    data = data.to(DEVICE)
                    output = net(data).to(DEVICE)
                    pred = torch.argmax(output, 1).to('cpu')
                    correct += (pred == label).sum().float().item()
                    total += len(label)
                acc = correct / total
                print('Accuracy on test:' + str(acc))
                correct = 0
                total = 0
                for data, label in tqdm(train_loader):
                    data = data.to(DEVICE)
                    output = net(data).to(DEVICE)
                    pred = torch.argmax(output, 1).to('cpu')
                    correct += (pred == label).sum().float().item()
                    total += len(label)
                acc = correct / total
                print('Accuracy on train:' + str(acc))

            net.train()

        torch.save(net.state_dict(), os.path.join(WEIGHT_DIR, 'weight.pkl'))

        if epoch % CHECKPOINT_SAVE_FREQ == 0:
            torch.save(net.state_dict(), os.path.join(
                WEIGHT_DIR, 'checkpoint_epoch#%d.pkl' % epoch))


if __name__ == "__main__":
    train()
