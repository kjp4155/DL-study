import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms


batch_size = 128
lr = 0.1


mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0, ), (1.0,))
])

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                            download=True, transform=mnist_transform)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True,transform=mnist_transform)

train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size)

model = torchvision.models.vgg.vgg11()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


epochs = 100
for epoch in range(epochs):
    for batch_idx, (data, answer) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.view(-1, 28 * 28)
        output = model(data)

        loss = F.nll_loss(output, answer)
        loss.backward()
        optimizer.step()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, answer) in enumerate(test_loader):
            data = data.view(-1, 28 * 28)
            output = model(data)
            test_loss += F.nll_loss(output, answer, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(answer.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('[Epoch {}/{}] Loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
        epoch, epochs,
        test_loss, correct, len(test_loader.dataset),
        correct / len(test_loader.dataset) * 100.0
    ))