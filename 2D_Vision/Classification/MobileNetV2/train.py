import numpy as np
import argparse

from datetime import datetime
from mobilenetV2_utill.acc_matrix import *
from mobilenetV2_utill.custom_data_load import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MobileNetV2_model.MobileNetV2 import *
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def train(train_loader, model, criterion, optimizer, device):
    '''
    training loop의 training 단계에 대한 함수
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # 순전파
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # 역전파
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    training loop의 validation 단계에 대한 함수
    '''

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # 순전파와 손실 기록하기
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    전체 training loop를 정의하는 함수
    '''

    # metrics를 저장하기 위한 객체 설정
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # VggNet_model 학습하기
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./images/', help='변환하고자 하는 이미지가 저장된 경로')
    parser.add_argument('--save_dir', type=str, default='./results/', help='변환한 이미지를 저장하고자 하는 경로')
    parser.add_argument('--classes', type=int, default=10, help='class 개수')
    parser.add_argument('--random_seed', type=int, default=66, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning_rate default 0.001')
    parser.add_argument('--epoch', type=int, default=10, help='num epochs default 30')
    parser.add_argument('--batch', type=int, default=32, help='batch_size default 32')
    parser.add_argument('--img_size', type=int, default=32, help='resize size default 32')


    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    # parameters
    RANDOM_SEED = opt.random_seed
    LEARNING_RATE = opt.lr
    BATCH_SIZE = opt.batch
    N_EPOCHS = opt.epoch

    IMG_SIZE = opt.img_size
    N_CLASSES = opt.classes


    # transforms 정의하기

    transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])

    # data set 다운받고 생성하기
    train_dataset = datasets.MNIST(root='mnist_data',
                                   train=True,
                                   transform=transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root='mnist_data',
                                   train=False,
                                   transform=transforms)

    # data loader 정의하기
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
    torch.manual_seed(RANDOM_SEED)

    model = LeNet_5(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader,
                                        valid_loader, N_EPOCHS, DEVICE)