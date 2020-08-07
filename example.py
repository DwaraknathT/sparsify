import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from Pruner import Pruner
from models.resnet import resnet20

# hyper params
batch_size = 100
epochs = 200


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = resnet20(10)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, 0, 0.1,
                                             step_size_up=50000, step_size_down=50000)
pruner = Pruner(
  net, optimizer, 0, 0.95, lr_scheduler=lr_sched,
  total_steps=(batch_size * len(trainloader)), ramping=True
)

for epoch in range(epochs):  # loop over the dataset multiple times
  # pruner.mask_sparsity()
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = pruner.model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    # optimizer.step()
    pruner.step()

    # print statistics
    running_loss += loss.item()
  print('[%d, %5d] loss: %.3f lr: %.3f' %
        (epoch + 1, i + 1, running_loss / 500, get_lr(pruner.optimizer)))
  running_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      inputs, labels = data[0].to(device), data[1].to(device)
      outputs = pruner.model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network on testset: %d %%' % (
      100 * correct / total))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = pruner.model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on testset: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
  for data in testloader:
    inputs, labels = data[0].to(device), data[1].to(device)
    outputs = pruner.model(inputs)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1

for i in range(10):
  print('Accuracy of %5s : %2d %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]))
