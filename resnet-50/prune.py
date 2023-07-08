import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from Branchy import Branchy_ResNet

import time

# gpu info
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(42)
else:
    print('Not using CUDA!!!')


# Load Pre-trained VGG-16 Model
model = torch.load('saves/initial_res50_model_ee.ptmodel').to(device)


def prune_weight(weight, threshold):
    weight_dev = weight.device
    tensor = weight.data.cpu().numpy()
    mask = torch.ones(tensor.shape).data.cpu().numpy()
    mask = np.where(abs(tensor) < threshold, 0, mask)
    print(f"Number of weights alive in this layer: {mask.sum()}/{mask.size}")
    return torch.from_numpy(tensor * mask).to(weight_dev)


def prune_model(model, s=1.5):
    for name, module in model.named_modules():
        if name in ['fc.0']:
            threshold = np.std(module.weight.data.cpu().numpy()) * s
            print(f'Pruning with threshold : {threshold} for layer {name}')
            module.weight.data = prune_weight(module.weight, threshold)


prune_model(model)
model.to(device)

# for name, module in model.named_modules():
#     if name in ['classifier.0', 'classifier.3', 'classifier.6']:
#         print(module.weight.data.cpu().numpy())


kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

############### B-ResNet-50 Test on CIFAR-100 #######################

test_data = torchvision.datasets.CIFAR100(root='../Pycharm_Project_1/datasets/',
                                          train=False,
                                          transform=transform,
                                          download=True)

test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=1,
                                           **kwargs)



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

# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.data.view_as(pred)).sum().item()
#
#         test_loss /= len(test_loader.dataset)
#         accuracy = 100. * correct / len(test_loader.dataset)
#         print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
#     return accuracy

# Initial training
print("--- After Pruning ---")
t1 = time.time()
accuracy = test()
t2 = time.time()
print(f"initial_accuracy {accuracy}")
print(f"Running time with early exit: {(t2 - t1) * 1000} ms")
torch.save(model, f"saves/prune_res50_model_ee.ptmodel")
torch.save(model.state_dict(), f"saves/prune_res50_param_ee.ptmodel")


