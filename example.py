import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.resnet import resnet20
import torchvision
import torchvision.transforms as transforms
from Pruner import Pruner

from layers.layers import MaskedConv, MaskedDense

#hyper params
batch_size = 100
epochs = 100

transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = MaskedConv(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = MaskedConv(6, 16, 5)
    self.fc1 = MaskedDense(16 * 5 * 5, 120)
    self.fc2 = MaskedDense(120, 84)
    self.fc3 = MaskedDense(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


net = resnet20(10)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
pruner = Pruner(
  net, optimizer, 0, 0.95, total_steps=(batch_size * len(trainloader)), ramping=True)


for epoch in range(epochs):  # loop over the dataset multiple times
  #pruner.mask_sparsity()
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
    #optimizer.step()
    pruner.step()

    # print statistics
    running_loss += loss.item()
  print('[%d, %5d] loss: %.3f' %
        (epoch + 1, i + 1, running_loss / 500))
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
