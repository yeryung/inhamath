!pip install efficientnet_pytorch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn, optim, cuda, from_numpy, save, rand, div, load
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import datetime
import glob


# get the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


rel_path = "/content/drive/MyDrive/cifar-100-python/"
test_row_data = unpickle(rel_path + "test")


class CustomDataset():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        return self.images[ind], self.labels[ind]


test_data = test_row_data[b'data'].reshape(10000, 3, 32, 32)
test_data = from_numpy(test_data).float()
test_dataset = CustomDataset(test_data, test_row_data[b'fine_labels'])

# Data loader, dataset 을 pythorch 에서 사용하기 편한형태로 변환
batch_size = 256

test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

lossfunction = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # stochastic gradient descent

device = "cuda" if cuda.is_available() else "cpu"  # device가 cuda 면 행렬연산에서 cpu 보다 빠른연산을 수행할 수 있는 gpu(그래픽카드)를 사용한다. 그렇지 않으면 그냥 cpu 를 사용한다
print(f"Training CIFAR on '{device}'\n{'=' * 44}")  # cpu 를 사용하는지 gpu(cuda는 nvidia)를 사용하는지 출력해줌

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100).to(device)
# summary(model, input_size=(3, 32, 32), device=device)
weight_list = glob.glob("/content/drive/MyDrive/CIFAR-100_weight/*")
for weight in weight_list:
    model.load_state_dict(load(weight))
    model.eval()  # nn.Module.eval method, train 과 test 를 구분, train 할 때는 model.train 을 호출한다
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(
            device)  # gpu 에 tensor 들을 할당함(cpu가 아닌 gpu 로 계산하기 위함), 여기서 data, target 은 tensor 이다
        output = model(data)

        # sum up batch loss
        test_loss += lossfunction(output, target).item()

        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print("=" * 44)
    print(
        f"Test set: Average loss : {test_loss:.4f}, Accuracy : {correct}/{len(test_loader.dataset)}({100. * correct / len(test_loader.dataset):.0f}%)")