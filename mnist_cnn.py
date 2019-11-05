import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms

import argparse
import plotter
import sys, time


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        conv1 = nn.Conv2d(1, 20, 5, stride=1)
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(20, 50, 5, stride=1)
        pool2 = nn.MaxPool2d(2)

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(), pool1,
            conv2, nn.ReLU(), pool2,
        )

        fc1 = nn.Linear(4 * 4 * 50, 500)
        fc2 = nn.Linear(500, 10)

        self.fc_module = nn.Sequential(
            fc1, nn.ReLU(),
            fc2
        )

    def forward(self, x):
        x = self.conv_module(x)
        x = x.view(len(x), -1)
        x = self.fc_module(x)
        return F.log_softmax(x, dim=1)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, (100. * correct / len(test_loader.dataset))

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return train_loss / len(train_loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        print('GPUs available.')


    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model =  CNNModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
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

        plotter.draw_loss(iters, train_losses, test_losses, accuracies)


if __name__ == '__main__':
    main()