import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


# model = models.vgg16()
# print(models)
#
# model.classifier[0].type(torch.int8)
#
# def print_model_parameters(model, with_values=False):
#     print(f"{'Param name':20} {'Shape':30} {'Type':15}")
#     print('-'*70)
#     for name, param in model.named_parameters():
#         print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
#         if with_values:
#             print(param)
#
# print_model_parameters(model)


import numpy as np

# codebook_str = '111100001'
# # b = np.array([[1,1,3,3],[1,2,3,2],[0,2,3,1]])
# # shape = b.shape
# num_of_padding = -len(codebook_str) % 8
# print(len(codebook_str)%8)
# header = f"{num_of_padding:08b}"
# codebook_str = header + codebook_str + '0' * num_of_padding
#
# print(codebook_str)
acc = [10, 85.39, 85.39,85.39, 85.45,85.45, 85.45]
thresh = [0,1,2,3,4,5,6]
plt.figure()
plt.plot(thresh, acc, "b-", linewidth=2)
# plt.legend(loc="upper right")
plt.xlabel("Number of bits in shared weights")
plt.ylabel("Test Accuracy")
plt.title("Top-1 Accuracy vs. Number of bits per effective weight in FC layers")
plt.savefig("quant_acc.png")


# Param name           Shape                          Type
# ----------------------------------------------------------------------
# features.0.weight    torch.Size([64, 3, 3, 3])      torch.float32
# features.0.bias      torch.Size([64])               torch.float32
# features.2.weight    torch.Size([64, 64, 3, 3])     torch.float32
# features.2.bias      torch.Size([64])               torch.float32
# features.5.weight    torch.Size([128, 64, 3, 3])    torch.float32
# features.5.bias      torch.Size([128])              torch.float32
# features.7.weight    torch.Size([128, 128, 3, 3])   torch.float32
# features.7.bias      torch.Size([128])              torch.float32
# features.10.weight   torch.Size([256, 128, 3, 3])   torch.float32
# features.10.bias     torch.Size([256])              torch.float32
# features.12.weight   torch.Size([256, 256, 3, 3])   torch.float32
# features.12.bias     torch.Size([256])              torch.float32
# features.14.weight   torch.Size([256, 256, 3, 3])   torch.float32
# features.14.bias     torch.Size([256])              torch.float32
# features.17.weight   torch.Size([512, 256, 3, 3])   torch.float32
# features.17.bias     torch.Size([512])              torch.float32
# features.19.weight   torch.Size([512, 512, 3, 3])   torch.float32
# features.19.bias     torch.Size([512])              torch.float32
# features.21.weight   torch.Size([512, 512, 3, 3])   torch.float32
# features.21.bias     torch.Size([512])              torch.float32
# features.24.weight   torch.Size([512, 512, 3, 3])   torch.float32
# features.24.bias     torch.Size([512])              torch.float32
# features.26.weight   torch.Size([512, 512, 3, 3])   torch.float32
# features.26.bias     torch.Size([512])              torch.float32
# features.28.weight   torch.Size([512, 512, 3, 3])   torch.float32
# features.28.bias     torch.Size([512])              torch.float32
# classifier.0.weight  torch.Size([4096, 25088])      torch.float32
# classifier.0.bias    torch.Size([4096])             torch.float32
# classifier.3.weight  torch.Size([4096, 4096])       torch.float32
# classifier.3.bias    torch.Size([4096])             torch.float32
# classifier.6.weight  torch.Size([1000, 4096])       torch.float32
# classifier.6.bias    torch.Size([1000])             torch.float32
# classifier.added_linear.weight torch.Size([10, 1000])         torch.float32
# classifier.added_linear.bias torch.Size([10])               torch.float32




# Param name           Shape                          Type
# ----------------------------------------------------------------------
# features.0.weight    torch.Size([876])              torch.uint8
# features.0.bias      torch.Size([64])               torch.float32
# features.0.codebook  torch.Size([33])               torch.float64
# features.0.hftree    torch.Size([137])              torch.uint8
# features.0.shape     torch.Size([4])                torch.int64
# features.2.weight    torch.Size([17997])            torch.uint8
# features.2.bias      torch.Size([64])               torch.float32
# features.2.codebook  torch.Size([33])               torch.float64
# features.2.hftree    torch.Size([137])              torch.uint8
# features.2.shape     torch.Size([4])                torch.int64
# features.5.weight    torch.Size([37688])            torch.uint8
# features.5.bias      torch.Size([128])              torch.float32
# features.5.codebook  torch.Size([33])               torch.float64
# features.5.hftree    torch.Size([137])              torch.uint8
# features.5.shape     torch.Size([4])                torch.int64
# features.7.weight    torch.Size([78122])            torch.uint8
# features.7.bias      torch.Size([128])              torch.float32
# features.7.codebook  torch.Size([33])               torch.float64
# features.7.hftree    torch.Size([137])              torch.uint8
# features.7.shape     torch.Size([4])                torch.int64
# features.10.weight   torch.Size([153944])           torch.uint8
# features.10.bias     torch.Size([256])              torch.float32
# features.10.codebook torch.Size([33])               torch.float64
# features.10.hftree   torch.Size([137])              torch.uint8
# features.10.shape    torch.Size([4])                torch.int64
# features.12.weight   torch.Size([317655])           torch.uint8
# features.12.bias     torch.Size([256])              torch.float32
# features.12.codebook torch.Size([33])               torch.float64
# features.12.hftree   torch.Size([137])              torch.uint8
# features.12.shape    torch.Size([4])                torch.int64
# features.14.weight   torch.Size([311903])           torch.uint8
# features.14.bias     torch.Size([256])              torch.float32
# features.14.codebook torch.Size([33])               torch.float64
# features.14.hftree   torch.Size([137])              torch.uint8
# features.14.shape    torch.Size([4])                torch.int64
# features.17.weight   torch.Size([630967])           torch.uint8
# features.17.bias     torch.Size([512])              torch.float32
# features.17.codebook torch.Size([33])               torch.float64
# features.17.hftree   torch.Size([137])              torch.uint8
# features.17.shape    torch.Size([4])                torch.int64
# features.19.weight   torch.Size([1292905])          torch.uint8
# features.19.bias     torch.Size([512])              torch.float32
# features.19.codebook torch.Size([33])               torch.float64
# features.19.hftree   torch.Size([137])              torch.uint8
# features.19.shape    torch.Size([4])                torch.int64
# features.21.weight   torch.Size([1292259])          torch.uint8
# features.21.bias     torch.Size([512])              torch.float32
# features.21.codebook torch.Size([33])               torch.float64
# features.21.hftree   torch.Size([137])              torch.uint8
# features.21.shape    torch.Size([4])                torch.int64
# features.24.weight   torch.Size([1307636])          torch.uint8
# features.24.bias     torch.Size([512])              torch.float32
# features.24.codebook torch.Size([33])               torch.float64
# features.24.hftree   torch.Size([137])              torch.uint8
# features.24.shape    torch.Size([4])                torch.int64
# features.26.weight   torch.Size([1319107])          torch.uint8
# features.26.bias     torch.Size([512])              torch.float32
# features.26.codebook torch.Size([33])               torch.float64
# features.26.hftree   torch.Size([137])              torch.uint8
# features.26.shape    torch.Size([4])                torch.int64
# features.28.weight   torch.Size([1327451])          torch.uint8
# features.28.bias     torch.Size([512])              torch.float32
# features.28.codebook torch.Size([33])               torch.float64
# features.28.hftree   torch.Size([137])              torch.uint8
# features.28.shape    torch.Size([4])                torch.int64
# classifier.0.codebook torch.Size([3])                torch.float64
# classifier.0.bias    torch.Size([4096])             torch.float32
# classifier.0.weight  torch.Size([14520885])         torch.uint8
# classifier.0.hftree  torch.Size([14])               torch.uint8
# classifier.3.codebook torch.Size([3])                torch.float64
# classifier.3.bias    torch.Size([4096])             torch.float32
# classifier.3.weight  torch.Size([2381761])          torch.uint8
# classifier.3.hftree  torch.Size([14])               torch.uint8
# classifier.6.codebook torch.Size([3])                torch.float64
# classifier.6.bias    torch.Size([1000])             torch.float32
# classifier.6.weight  torch.Size([579975])           torch.uint8
# classifier.6.hftree  torch.Size([14])               torch.uint8
# classifier.added_linear.weight torch.Size([10, 1000])         torch.float32
# classifier.added_linear.bias torch.Size([10])               torch.float32

#
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#     (added_linear): Linear(in_features=1000, out_features=10, bias=True)
#     (softmax): Softmax(dim=1)
#   )
# )
