import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from Branchy import Branchy_VGG
import time


# model = models.vgg16(weights='IMAGENET1K_V1')
# torch.save(model.state_dict(), 'model_weights.pth')

# gpu info
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(42)
else:
    print('Not using CUDA!!!')


# Initialize VGG-16 model
model = models.vgg16() # Untrained Weights
model.load_state_dict(torch.load('model_weights.pth')) # Load Pre-trained Weights

# Adapt to 10-category classification (CIFAR-10)
model.classifier.add_module('added_linear', nn.Linear(1000,10))
model.classifier.add_module('softmax', nn.Softmax(dim=1))


# Model Structure (Built-in output)
print(model)

# Model Parameter (Self-defined output)
def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


print_model_parameters(model)


# Add Branches
model = Branchy_VGG(model)
model.to(device)

# Model Structure (Built-in output)
print(model)

# Model Parameter (Self-defined output)
def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


print_model_parameters(model)


############### Branchy VGG-16 Test on CIFAR-10 #######################
kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root='/public/torchvision_datasets/',
                                          train=True,
                                          transform=transform,
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root='/public/torchvision_datasets/',
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


optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
initial_optimizer_state_dict = optimizer.state_dict()
loss_fn = nn.CrossEntropyLoss()


def train(epochs):
    model.train()
    acc = []

    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            model.set_training_mode()
            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output0, output1 = model(data)
            loss = 1.0 * loss_fn(output0, target) + 0.3 * loss_fn(output1, target)
            loss.backward()

            optimizer.step()
            if batch_idx % 10 == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
        accuracy = test()
        acc.append(accuracy)
    plt.figure()
    plt.plot(acc, "r-+", linewidth=2)
    # plt.legend(loc="upper right")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy of VGG-16 on CIFAR-10 Dataset")
    plt.savefig("vgg-acc.png")
    # print(f"test ccuracy: {accuracy}")


def test():
    model.eval()
    model.set_inference_mode()
    test_loss = 0
    correct = 0
    num_exit = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, flag = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            num_exit += flag

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        percent_exit = 100. * num_exit / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), Percent of Early Exit cases: {percent_exit: .2f}%')
    return accuracy


# Initial training
print("--- Initial training ---")
train(20)
t1 = time.time()
accuracy = test()
t2 = time.time()
print(f"initial_accuracy {accuracy}")
print(f'Running time with early exit: {(t2 - t1) * 1000} ms')
torch.save(model, f"saves/initial_vgg16_model.ptmodel")
torch.save(model.state_dict(), f"saves/initial_vgg16_param.ptmodel")

