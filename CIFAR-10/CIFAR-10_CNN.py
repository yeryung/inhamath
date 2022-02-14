import pickle
import numpy as np
from __future__ import print_function
from torch import nn, optim, cuda, from_numpy, save
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
from itertools import chain

# get the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

name_path = "/content/drive/MyDrive/CIFAR-10/batches.meta"
name_map = unpickle(name_path)[b'label_names']

train_paths = [f"/content/drive/MyDrive/CIFAR-10/data_batch_{i}" for i in range(1, 6)] 
tmp = tuple(map(lambda path: unpickle(path), train_paths))
train_images = np.concatenate(tuple(map(lambda batch: np.reshape(batch[b"data"], (10000, 3, 32, 32)), tmp)))
train_images = from_numpy(train_images).float()
train_labels = list(chain.from_iterable(map(lambda batch: batch[b'labels'], tmp)))

test_path = "/content/drive/MyDrive/CIFAR-10/test_batch"
tmp = unpickle(test_path)
test_images = from_numpy(np.reshape(tmp[b"data"], (10000, 3, 32, 32))).float()
test_labels = tmp[b'labels']

class CustomDataset():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        return self.images[ind], self.labels[ind]

train_dataset = CustomDataset(train_images, train_labels)
test_dataset = CustomDataset(test_images, test_labels)

#Data loader, dataset 을 pythor 에서 사용하기 편한형태로 변환
batch_size = 32
train_loader = data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)
test_loader = data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)

# design network architecture
class Net(nn.Module): #nn.Module class 를 상속
    def __init__(self):
        super(Net, self).__init__() #nn.Module 의 __init__ 을 호출
        self.l1 = nn.Conv2d(
            in_channels = 3, # input tensor 의 depth
            out_channels = 16, # filter 의 개수
            kernel_size = 3, # filter 의 width, height 
            padding = 1
            )
        self.l2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.l3 = nn.Conv2d(16, 32, 3, padding = 1)
        self.l4 = nn.Conv2d(32, 32, 3, padding = 1)
        self.l5 = nn.Conv2d(32, 32, 3, padding = 1)
        self.l6 = nn.Linear(2048, 128)
        self.l7 = nn.Linear(128, 10) 

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = x.view(-1, 2048)
        x = F.relu(self.l6(x))
        return self.l7(x)

device = "cuda" if cuda.is_available() else "cpu" # device가 cuda 면 행렬연산에서 cpu 보다 빠른연산을 수행할 수 있는 gpu(그래픽카드)를 사용한다. 그렇지 않으면 그냥 cpu 를 사용한다
print(f"Training CIFAR on '{device}'\n{'='*44}") # cpu 를 사용하는지 gpu(cuda는 nvidia)를 사용하는지 출력해줌

model = Net()
model.to(device)

lossfunction = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5) # stochastic gradient descent

def train(epoch):
    model.train() # nn.Module.train method, train 과 test 를 구분, test 할 때는 modle.eval 을 호출한다
    for batch_ind, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad() # gradients 값을 0으로 초기화
        output = model(data)
        loss = lossfunction(output, target)
        loss.backward()
        optimiser.step()
        if batch_ind % 100 == 0:
            print('Train Epoch : {} | Batch Status : {}/{} ({:.0f}%) | Loss : {:.6f}'.format(
                epoch, 
                batch_ind*len(data), len(train_loader.dataset), 100. * batch_ind / len(train_loader),
                loss.item()
                ))

def test():
    model.eval() # nn.Module.eval method, train 과 test 를 구분, train 할 때는 model.train 을 호출한다
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device) # gpu 에 tensor 들을 할당함(cpu가 아닌 gpu 로 계산하기 위함), 여기서 data, target 은 tensor 이다
        output = model(data)

        # sum up batch loss
        test_loss += lossfunction(output, target).item()

        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print("="*44)
    print(f"Test set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}({100. * correct / len(test_loader.dataset):.0f}%)")


# for epoch in range(1, 10):
#     train(epoch)
#     save_path = f"/content/drive/MyDrive/CIFAR-10/param/{epoch:0>3}"
#     save(model.state_dict(), save_path)
#     test()