import numpy as np
import argparse

from datetime import datetime
from ResneXt_utill.acc_matrix import *
from ResneXt_utill.CustomDataset import *
from ResNeXt_model.ResNeXt import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim


import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/data/time_series_data/베어링/cwru/Convert_img/12k_DE/classification',
                        help='변환하고자 하는 이미지가 저장된 경로')
    parser.add_argument('--save_dir', type=str, default='./results/', help='변환한 이미지를 저장하고자 하는 경로')
    parser.add_argument('--random_seed', type=int, default=66, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate default 0.001')
    parser.add_argument('--epoch', type=int, default=10, help='num epochs default 30')
    parser.add_argument('--batch', type=int, default=8, help='batch_size default 32')
    parser.add_argument('--img_size', type=int, default=224, help='resize size default 32')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    # parameters
    RANDOM_SEED = opt.random_seed
    IMG_SIZE = opt.img_size

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    hyper_param_epoch = opt.epoch
    hyper_param_batch = opt.batch
    hyper_param_learning_rate = opt.lr

    transforms_train = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                           transforms.RandomRotation(10.),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                          transforms.ToTensor()])

    train_data_set = CustomImageDataset(data_set_path=f"{opt.root_dir}/test", transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    validation_data_set = CustomImageDataset(data_set_path=f"{opt.root_dir}/test", transforms=transforms_test)
    validation_loader = DataLoader(validation_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path=f"{opt.root_dir}/test", transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

    if not (train_data_set.num_classes == validation_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")
        exit()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = train_data_set.num_classes

    # custom_model = SE_ResNext(n_classes=num_classes).to(device)
    custom_model = resnetX50(n_classes=num_classes).to(device)
    # vgg_pre = models.vgg16(pretrained=True)
    state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'], progress=True)
    custom_model.load_state_dict(state_dict, strict=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    best_acc = 0.
    for e in range(hyper_param_epoch):
        train_losses = []
        val_losses = []
        for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = custom_model(images)

            loss = criterion(outputs, labels)
            train_losses.append(loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % hyper_param_batch == 0:
                print('Epoch [{}/{}], Train Step [{}/{}] ,Loss: {:.4f}'.format(e + 1, hyper_param_epoch, i_batch,
                                                                               len(train_loader), loss.item()))
        # validation the model
        custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for item in validation_loader:
                images = item['image'].to(device)
                labels = item['label'].to(device)
                outputs = custom_model(images)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss)
                _, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += (predicted == labels).sum().item()

            print('Epoch [{}/{}], Loss: {:.4f}, val_Loss: {:.4f}'.format(e + 1, hyper_param_epoch,
                                                                         sum(train_losses) / len(train_losses),
                                                                         sum(val_losses) / len(val_losses)))
        scheduler.step()
        if best_acc < (100 * correct / total):
            best_acc = (100 * correct / total)
            torch.save(custom_model, f'{round((100 * correct / total),2)}_ResNeXtnet.pth')

    # # validation the model
    # custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for item in test_loader:
    #         images = item['image'].to(device)
    #         labels = item['label'].to(device)
    #         outputs = custom_model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += len(labels)
    #         correct += (predicted == labels).sum().item()
    #
    #     print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
