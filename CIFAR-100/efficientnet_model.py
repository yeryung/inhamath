import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn, optim, cuda, from_numpy, save, rand, div
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import datetime

# get the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

rel_path = "/content/drive/MyDrive/cifar-100-python/"
train_row_data = unpickle(rel_path + "train")
test_row_data = unpickle(rel_path + "test")
meta_row_data = unpickle(rel_path + "meta")

class CustomDataset():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        return self.images[ind], self.labels[ind]


train_data = train_row_data[b'data'].reshape(50000, 3, 32, 32)
train_data = from_numpy(train_data).float()

test_data = test_row_data[b'data'].reshape(10000, 3, 32, 32)
test_data = from_numpy(test_data).float()

train_dataset = CustomDataset(train_data, train_row_data[b'fine_labels'])
test_dataset = CustomDataset(test_data, test_row_data[b'fine_labels'])

# Data loader, dataset 을 pythorch 에서 사용하기 편한형태로 변환
batch_size = 32
train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

device = "cuda" if cuda.is_available() else "cpu"  # device가 cuda 면 행렬연산에서 cpu 보다 빠른연산을 수행할 수 있는 gpu(그래픽카드)를 사용한다. 그렇지 않으면 그냥 cpu 를 사용한다
print(f"Training CIFAR on '{device}'\n{'=' * 44}")  # cpu 를 사용하는지 gpu(cuda는 nvidia)를 사용하는지 출력해줌

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100).to(device)
# summary(model, input_size=(3, 32, 32), device=device)

lossfunction = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # stochastic gradient descent

def train(epoch):
    model.train()  # nn.Module.train method, train 과 test 를 구분, test 할 때는 modle.eval 을 호출한다
    for batch_ind, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()  # gradients 값을 0으로 초기화
        output = model(data)
        loss = lossfunction(output, target)
        loss.backward()
        optimiser.step()
        if batch_ind % 100 == 0:
            print('Train Epoch : {} | Batch Status : {}/{} ({:.0f}%) | Loss : {:.6f}'.format(
                epoch,
                batch_ind * len(data), len(train_loader.dataset), 100. * batch_ind / len(train_loader),
                loss.item()
            ))

rel_save_path = "/content/drive/MyDrive/CIFAR-100_weight/"
for epoch in range(1, 2):
    train(epoch)
    serial = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {batch_size} {epoch}"
    save(model.state_dict(), rel_save_path + serial)