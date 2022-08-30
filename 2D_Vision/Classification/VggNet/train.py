import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from VggNet_utill.CustomDataset import CustomImageDataset
from VggNet_model.VggNet import VGGNet
import argparse
import torch.optim as optim



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./images/', help='변환하고자 하는 이미지가 저장된 경로')
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

    train_data_set = CustomImageDataset(data_set_path="/data/time_series_data/베어링/cwru/Convert_img/12k_DE/classification/train", transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    validation_data_set = CustomImageDataset(data_set_path="/data/time_series_data/베어링/cwru/Convert_img/12k_DE/classification/validation", transforms=transforms_test)
    validation_loader = DataLoader(validation_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path="/data/time_series_data/베어링/cwru/Convert_img/12k_DE/classification/test", transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)
    
    if not (train_data_set.num_classes == validation_data_set.num_classes):
        print("error: Numbers of class in training set and test set are not equal")
        exit()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = train_data_set.num_classes
    custom_model = VGGNet(n_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    for e in range(hyper_param_epoch):
        train_losses = []
        val_losses = []
        for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = custom_model(images)

            loss= criterion(outputs, labels)
            train_losses.append(loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % hyper_param_batch == 0:
                print('Epoch [{}/{}], Train Step [{}/{}] ,Loss: {:.4f}'.format(e + 1, hyper_param_epoch, i_batch, len(train_loader), loss.item()))
        # validation the model
        custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            for item in validation_loader:
                images = item['image'].to(device)
                labels = item['label'].to(device)
                outputs = custom_model(images)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss)
            print('Epoch [{}/{}], Loss: {:.4f}, val_Loss: {:.4f}'.format(e + 1, hyper_param_epoch, sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)))
        scheduler.step()
        
    # validation the model
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
