import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms

import argparse
import plotter, vgg, modified_vgg
import sys, time

def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.float()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, (100. * correct / len(test_loader.dataset))

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    criterion = nn.CrossEntropyLoss()

    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.float() # sum up batch loss
        loss.backward()
        optimizer.step()

        if args.log_interval != -1 and batch_idx % args.log_interval == 0 :
            print('\r\033[K', end='')
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.float() / args.batch_size), end='')

    return train_loss / len(train_loader.dataset)

def load_data(args):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_list = []
    if args.data_aug:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomCrop(32, 4))
    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose(transform_list)),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def load_model(args):
    if args.model == 'vgg11':
        model = vgg.vgg11().to(device)
    if args.model == 'vgg13':
        model = vgg.vgg13().to(device)
    if args.model == 'vgg16':
        model = vgg.vgg16().to(device)
    elif args.model == 'vgg19':
        model = vgg.vgg19().to(device)
    elif args.model == 'modified_vgg11':
        model = modified_vgg.vgg11().to(device)
    elif args.model == 'modified_vgg13':
        model = modified_vgg.vgg13().to(device)
    elif args.model == 'modified_vgg16':
        model = modified_vgg.vgg16().to(device)
    elif args.model == 'modified_vgg19':
        model = modified_vgg.vgg19().to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=12, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--plot-name', type=str, default='test')
    parser.add_argument('--model', type=str, default='vgg11')
    parser.add_argument("--data-aug", action="store_true", default=False)

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        print('GPUs available.')

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader, test_loader = load_data(args)
    model = load_model(args)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    iters, test_losses, train_losses, accuracies = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(args, model, device, test_loader)

        end_time = time.time()

        print('Elapsed time: {}'.format(end_time - start_time))

        iters.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        plotter.draw_loss(args.plot_name, iters, train_losses, test_losses, accuracies)


if __name__ == '__main__':
    main()