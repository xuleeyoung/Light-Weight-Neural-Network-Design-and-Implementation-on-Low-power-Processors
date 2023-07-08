import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import Parameter
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

from Branchy import Branchy_VGG
from quant import quant_Linear
import quant
import time


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(42)
else:
    print('Not using CUDA!!!')


# Load Pre-trained VGG-16 Model
model = torch.load('saves/prune_res50_model_ee.ptmodel').to(device)
print(model)
def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

print_model_parameters(model)


### Quantization for FC layers
model.fc[0] = quant_Linear(model.fc[0])
model.exit1[0] = quant_Linear(model.exit1[0])
model.exit2[0] = quant_Linear(model.exit2[0])


### Quantization for Conv layers
quant.quant_Conv(model)


model.to(device)
print(model)
print_model_parameters(model)


torch.save(model, f"saves/quant_res50_model_ee.ptmodel")
torch.save(model.state_dict(), f"saves/quant_res50_param_ee.ptmodel")



### De-Quantization
quant.dequant(model)
model.fc[0].dequantize()
model.exit1[0].dequantize()
model.exit2[0].dequantize()


kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

############### Branchy VGG-16 Test on CIFAR-10 #######################

test_data = torchvision.datasets.CIFAR10(root='/public/torchvision_datasets/',
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


# Initial training
print("--- After Quantization ---")
t1 = time.time()
accuracy = test()
t2 = time.time()
print(f"initial_accuracy {accuracy}")
print(f"Running time with early exit: {(t2 - t1) * 1000} ms")
