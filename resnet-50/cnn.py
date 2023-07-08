import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time

# model = models.resnet50(weights='IMAGENET1K_V1')
# torch.save(model.state_dict(), 'model_weights.pth')

# gpu info
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(42)
else:
    print('Not using CUDA!!!')


# Initialize ResNet-50 model
model = torch.load('saves/initial_res18_model.ptmodel').to(device)
model.fc = nn.Sequential(
    nn.Linear(2048, 100),
    nn.Softmax(dim=1)
)
model.to(device)
#
#
#
print(model)
#
#
def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

print_model_parameters(model)
#
#


kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(15),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

############### ResNet-50 Test on CIFAR-100 #######################
train_data = torchvision.datasets.CIFAR100(root='../Pycharm_Project_1/datasets/',
                                          train=True,
                                          transform=transform,
                                          download=True)

test_data = torchvision.datasets.CIFAR100(root='../Pycharm_Project_1/datasets/',
                                          train=False,
                                          transform=transform,
                                          download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("The size of training set is {}.".format(train_data_size))
print("The size of training set is {}.".format(test_data_size))

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=128,
                                           **kwargs)

test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=1,
                                           **kwargs)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
initial_optimizer_state_dict = optimizer.state_dict()
loss_fn = nn.CrossEntropyLoss()


def train(epochs):
    model.train()
    acc = []

    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()
            if batch_idx % 10 == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
        # accuracy = test()
        # acc.append(accuracy)
    # plt.figure()
    # plt.plot(acc, "r-+", linewidth=2)
    # plt.legend(loc="upper right")
    # plt.xlabel("Number of Epochs")
    # plt.ylabel("Test Accuracy")
    # plt.title("Test Accuracy of Resnet-50 on CIFAR-100 Dataset")
    # plt.savefig("resnet-acc.png")
    # print(f"test ccuracy: {accuracy}")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
train(50)
t1 = time.time()
accuracy = test()
t2 = time.time()
print(f"initial_accuracy {accuracy}")
print(f'Running time without early exit: {(t2 - t1) * 1000} ms')
torch.save(model, f"saves/initial_res50_model_without.ptmodel")
torch.save(model.state_dict(), f"saves/initial_res50_param_without.ptmodel")

