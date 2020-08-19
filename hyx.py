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

CELEBA_LABEL_PATH = r'E:\GitHub项目\Dataset\celeba-dataset\list_attr_celeba.csv'
CELEBA_DATA_PATH = r'E:\GitHub项目\Dataset\celeba-dataset\img_align_celeba'
DEVICE = 'cuda'

ATTR_NAME = 'Goatee'
N_POS_SAMPLE = 10000  # 正样本数量
N_NEG_SAMPLE = 10000  # 负样本数量
TRAINVAL_RATIO = 10  # 验证集比例

BATCH_SIZE = 128
START_EPOCH = 1
TOTAL_EPOCH = 20
CONTINUE = False
LR = 0.01
TEST_FREQ = 5
ACC_THRE = 0.95


def make_data():
    if not os.path.exists(ATTR_NAME):
        os.mkdir(ATTR_NAME)  # 以数字权限模式创建目录？

    celeba_label = pd.read_csv(CELEBA_LABEL_PATH)
    celeba_label_attr = list(celeba_label.keys())
    attr_idx = celeba_label_attr.index(ATTR_NAME)

    celeba_label = np.array(celeba_label)  # 提取成表格
    np.random.shuffle(celeba_label)  # 打乱行顺序

    pos_list = list()
    neg_list = list()
    n_pos = 0
    n_neg = 0
    for row in celeba_label:
        if n_pos >= N_POS_SAMPLE and n_neg >= N_NEG_SAMPLE:
            break
        if n_pos < N_POS_SAMPLE and row[attr_idx] == 1:
            n_pos += 1
            pos_list.append(row[0])
        elif n_neg < N_NEG_SAMPLE and row[attr_idx] == -1:
            n_neg += 1
            neg_list.append(row[0])

    random.shuffle(pos_list)
    random.shuffle(neg_list)

    train_pos_set = pos_list[:(TRAINVAL_RATIO - 1) * N_POS_SAMPLE // TRAINVAL_RATIO]
    train_neg_set = neg_list[:(TRAINVAL_RATIO - 1) * N_NEG_SAMPLE // TRAINVAL_RATIO]
    val_pos_set = pos_list[(TRAINVAL_RATIO - 1) * N_POS_SAMPLE // TRAINVAL_RATIO:]
    val_neg_set = neg_list[(TRAINVAL_RATIO - 1) * N_NEG_SAMPLE // TRAINVAL_RATIO:]

    return train_pos_set, train_neg_set, val_pos_set, val_neg_set


class ImageDataset(Dataset):
    def __init__(self, pos_set, neg_set, mode='train'):
        super(ImageDataset, self).__init__()
        print('-------\nmake %s dataset!\n#pos_sample:%d\n#neg_sample:%d\n-------' % (mode, len(pos_set), len(neg_set)))
        self.image_list = pos_set + neg_set
        self.label_list = np.concatenate((np.ones(len(pos_set)), np.zeros(len(neg_set))), axis=0)
        if mode == 'train':
            self.transforms_ = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transforms_ = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
            ])
        # print(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(  # 加载图像：CELEBA_DATA_PATH：图像存放地址，idx：图像标签
            CELEBA_DATA_PATH, self.image_list[idx]))
        img = self.transforms_(img)  # 处理图像
        label = self.label_list[idx]
        label = torch.tensor(label, dtype=torch.long)
        return img, label


def get_loader(mode, pos_set, neg_set, shuffle, bs):
    data = ImageDataset(pos_set, neg_set, mode=mode)
    return DataLoader(data, batch_size=bs, shuffle=shuffle)


def get_resnet18():
    net = resnet18()
    modified_net = nn.Sequential(*list(net.children())[:-1])
    return modified_net


class ClassifyModel(nn.Module):
    def __init__(self, n_class=2):
        super(ClassifyModel, self).__init__()
        self.backbone = get_resnet18()
        self.extra_layer = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.extra_layer(out)
        return out


def train():
    pth = os.path.join(
        ATTR_NAME, 'weight.pkl')  # 路径拼接
    net = ClassifyModel().to(DEVICE)
    if CONTINUE:  # 检查是否有已经训练好的权重
        net.load_state_dict(torch.load(pth))
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    train_pos_set, train_neg_set, val_pos_set, val_neg_set = make_data()
    train_loader = get_loader(
        'train', train_pos_set, train_neg_set, shuffle=True, bs=BATCH_SIZE)
    test_loader = get_loader(
        'test', val_pos_set, val_neg_set, shuffle=False, bs=100)
    optimizer = optim.SGD(net.parameters(), lr=LR,
                          momentum=0.9, weight_decay=1e-4)
    start_time = datetime.datetime.now()

    for epoch in range(START_EPOCH, TOTAL_EPOCH + 1):
        for n_iter, (img, label) in enumerate(train_loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()

            predict_label = net(img)
            loss = criterion(predict_label, label)

            loss.backward()
            optimizer.step()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.2f] time: %s"
                % (
                    epoch,
                    TOTAL_EPOCH,
                    n_iter,
                    len(train_loader),
                    loss.item(),
                    datetime.datetime.now() - start_time
                )
            )



        if epoch % TEST_FREQ == 0:
            net.eval()
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

            if acc > ACC_THRE:
                break

            net.train()

        torch.save(net.state_dict(), pth)


def eval2():
    pth = os.path.join(
        ATTR_NAME, 'weight.pkl')
    net = ClassifyModel().to(DEVICE)
    net.load_state_dict(torch.load(pth))
    net.eval()

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    train_pos_set, train_neg_set, val_pos_set, val_neg_set = make_data()
    train_loader = get_loader(
        'train', train_pos_set, train_neg_set, shuffle=True, bs=BATCH_SIZE)
    with torch.no_grad():
        for data, label in tqdm(train_loader):
            data = data.to(DEVICE)
            output = net(data).to(DEVICE)
            pred = torch.argmax(output, 1).to('cpu')

            tp += ((pred == 1) & (label == 1)).sum().float().item()
            tn += ((pred == 0) & (label == 0)).sum().float().item()
            fp += ((pred == 1) & (label == 0)).sum().float().item()
            fn += ((pred == 0) & (label == 1)).sum().float().item()

    print('Accuracy     :   ' + str((tp + tn) / (tp + tn + fp + fn)))
    print('recall       :   ' + str((tp) / (tp + fn)))
    print('precision    :   ' + str((tp) / (tp + fp)))


if __name__ == "__main__":
    train()
    # eval2()
