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


class quant_Linear(nn.Module):
    def __init__(self, module):
        super(quant_Linear, self).__init__()
        self.shape = module.weight.data.shape
        # Bitwidth for weights
        self.bitwidth = 3
        # Codebook for quantization
        self.codebook = Parameter(torch.zeros(self.bitwidth), requires_grad=False)
        self.weight = Parameter(torch.zeros(self.shape, dtype=torch.uint8), requires_grad=False)
        self.bias = Parameter(module.bias.data)
        self.quantize(module)

    def quantize(self, module):
        weight_dev = self.weight.device
        codebook_dev = self.codebook.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2)
        print(mat.data.size)
        print("Start k means:")
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                        algorithm="full")
        kmeans.fit(mat.data.reshape(-1, 1))
        print("Finished k means:")
        mat.data = (kmeans.labels_ + 1).reshape(-1).astype(np.uint8)
        self.codebook.data = torch.from_numpy(np.concatenate(([0.], kmeans.cluster_centers_.reshape(-1)))).to(codebook_dev)
        self.weight.data = torch.from_numpy(mat.toarray()).to(weight_dev)

    def dequantize(self):
        dev = self.weight.device
        codebook = self.codebook.data.cpu().numpy()
        weight = self.weight.data.cpu().numpy()
        shape = weight.shape
        self.weight.data = torch.from_numpy(codebook[weight.reshape(-1)].reshape(shape)).type(torch.float32).to(dev)

    def forward(self, input):
        # dev = self.weight.device
        # codebook = self.codebook.data.cpu().numpy()
        # weight = self.weight.data.cpu().numpy()
        # shape = weight.shape
        # true_weight = torch.from_numpy(codebook[weight.reshape(-1)].reshape(shape)).type(torch.float32).to(dev)
        # # print(true_weight.data.cpu().numpy())
        return F.linear(input, self.weight, self.bias)

# Quantization for Convolutional Module (Conv Layers)
def quant_Conv(model):
    for name, module in model.named_modules():
        if 'conv' in name:
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            weight = weight.reshape(shape[0], shape[1] * shape[2] * shape[3])
            weight_dev = module.weight.device
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=32)
            print(mat.data.size)
            print("Start k means:")
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                            algorithm="full")
            kmeans.fit(mat.data.reshape(-1, 1))
            print("Finished k means:")
            mat.data = (kmeans.labels_ + 1).reshape(-1).astype(np.uint8)
            codebook = torch.from_numpy(np.concatenate(([0.], kmeans.cluster_centers_.reshape(-1)))).to(weight_dev)
            module.register_parameter('codebook', Parameter(codebook, requires_grad=False))
            module.weight = Parameter(torch.from_numpy(mat.toarray().reshape(shape)).to(weight_dev), requires_grad=False)


def dequant(model):
    for name, module in model.named_modules():
        if 'conv' in name:
            dev = module.weight.device
            codebook = module.codebook.data.cpu().numpy()
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            true_weight = torch.from_numpy(codebook[weight.reshape(-1)].reshape(shape)).type(torch.float32).to(dev)
            module.weight = Parameter(true_weight, requires_grad=False)



# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else 'cpu')
# if use_cuda:
#     print("Using CUDA!")
#     torch.cuda.manual_seed(42)
# else:
#     print('Not using CUDA!!!')
#
#
# # Load Pre-trained VGG-16 Model
# model = torch.load('saves/prune_vgg16_model.ptmodel').to(device)
# print(model)
# def print_model_parameters(model, with_values=False):
#     print(f"{'Param name':20} {'Shape':30} {'Type':15}")
#     print('-'*70)
#     for name, param in model.named_parameters():
#         print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
#         if with_values:
#             print(param)
#
# print_model_parameters(model)
#
#
# model.classifier[0] = quant_Linear(model.classifier[0])
# model.classifier[3] = quant_Linear(model.classifier[3])
# model.classifier[6] = quant_Linear(model.classifier[6])
# quant_Conv(model)
#
# model.to(device)
# print(model)
# print_model_parameters(model)
#
#
# torch.save(model, f"saves/quant_vgg16_model.ptmodel")
# torch.save(model.state_dict(), f"saves/quant_vgg16_param.ptmodel")
#
#
# dequant(model)
# kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
# ])
#
# ############### VGG-16 Test on CIFAR-10 #######################
#
# test_data = torchvision.datasets.CIFAR10(root='/public/torchvision_datasets/',
#                                           train=False,
#                                           transform=transform,
#                                           download=True)
#
# test_loader = torch.utils.data.DataLoader(test_data,
#                                            batch_size=128,
#                                            **kwargs)
#
#
#
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
#
# # Initial training
# print("--- After Quantization ---")
# accuracy = test()
# print(f"initial_accuracy {accuracy}")
#

