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
acc = [0.99 ,1.83, 24.04, 54.88, 59.11, 60.2, 60.43, 60.22]
thresh = [1, 2, 3, 4, 5, 6, 7, 8]
plt.figure()
plt.plot(thresh, acc, "b-", linewidth=2)
# plt.legend(loc="upper right")
plt.xlabel("Number of bits in shared weights")
plt.ylabel("Test Accuracy")
plt.title("Top-1 Accuracy vs. Number of bits per effective weight in Conv layers")
plt.savefig("quant_acc.png")


# conv1.codebook       torch.Size([33])               torch.float64
# conv1.hftree         torch.Size([137])              torch.uint8
# conv1.shape          torch.Size([4])                torch.int64
# bn1.weight           torch.Size([64])               torch.float32
# bn1.bias             torch.Size([64])               torch.float32
# layer1.0.conv1.weight torch.Size([1801])             torch.uint8
# layer1.0.conv1.codebook torch.Size([33])               torch.float64
# layer1.0.conv1.hftree torch.Size([137])              torch.uint8
# layer1.0.conv1.shape torch.Size([4])                torch.int64
# layer1.0.bn1.weight  torch.Size([64])               torch.float32
# layer1.0.bn1.bias    torch.Size([64])               torch.float32
# layer1.0.conv2.weight torch.Size([15875])            torch.uint8
# layer1.0.conv2.codebook torch.Size([33])               torch.float64
# layer1.0.conv2.hftree torch.Size([137])              torch.uint8
# layer1.0.conv2.shape torch.Size([4])                torch.int64
# layer1.0.bn2.weight  torch.Size([64])               torch.float32
# layer1.0.bn2.bias    torch.Size([64])               torch.float32
# layer1.0.conv3.weight torch.Size([6850])             torch.uint8
# layer1.0.conv3.codebook torch.Size([33])               torch.float64
# layer1.0.conv3.hftree torch.Size([137])              torch.uint8
# layer1.0.conv3.shape torch.Size([4])                torch.int64
# layer1.0.bn3.weight  torch.Size([256])              torch.float32
# layer1.0.bn3.bias    torch.Size([256])              torch.float32
# layer1.0.downsample.0.weight torch.Size([256, 64, 1, 1])    torch.float32
# layer1.0.downsample.1.weight torch.Size([256])              torch.float32
# layer1.0.downsample.1.bias torch.Size([256])              torch.float32
# layer1.1.conv1.weight torch.Size([7252])             torch.uint8
# layer1.1.conv1.codebook torch.Size([33])               torch.float64
# layer1.1.conv1.hftree torch.Size([137])              torch.uint8
# layer1.1.conv1.shape torch.Size([4])                torch.int64
# layer1.1.bn1.weight  torch.Size([64])               torch.float32
# layer1.1.bn1.bias    torch.Size([64])               torch.float32
# layer1.1.conv2.weight torch.Size([18229])            torch.uint8
# layer1.1.conv2.codebook torch.Size([33])               torch.float64
# layer1.1.conv2.hftree torch.Size([137])              torch.uint8
# layer1.1.conv2.shape torch.Size([4])                torch.int64
# layer1.1.bn2.weight  torch.Size([64])               torch.float32
# layer1.1.bn2.bias    torch.Size([64])               torch.float32
# layer1.1.conv3.weight torch.Size([7342])             torch.uint8
# layer1.1.conv3.codebook torch.Size([33])               torch.float64
# layer1.1.conv3.hftree torch.Size([137])              torch.uint8
# layer1.1.conv3.shape torch.Size([4])                torch.int64
# layer1.1.bn3.weight  torch.Size([256])              torch.float32
# layer1.1.bn3.bias    torch.Size([256])              torch.float32
# layer1.2.conv1.weight torch.Size([8431])             torch.uint8
# layer1.2.conv1.codebook torch.Size([33])               torch.float64
# layer1.2.conv1.hftree torch.Size([137])              torch.uint8
# layer1.2.conv1.shape torch.Size([4])                torch.int64
# layer1.2.bn1.weight  torch.Size([64])               torch.float32
# layer1.2.bn1.bias    torch.Size([64])               torch.float32
# layer1.2.conv2.weight torch.Size([18996])            torch.uint8
# layer1.2.conv2.codebook torch.Size([33])               torch.float64
# layer1.2.conv2.hftree torch.Size([137])              torch.uint8
# layer1.2.conv2.shape torch.Size([4])                torch.int64
# layer1.2.bn2.weight  torch.Size([64])               torch.float32
# layer1.2.bn2.bias    torch.Size([64])               torch.float32
# layer1.2.conv3.weight torch.Size([7340])             torch.uint8
# layer1.2.conv3.codebook torch.Size([33])               torch.float64
# layer1.2.conv3.hftree torch.Size([137])              torch.uint8
# layer1.2.conv3.shape torch.Size([4])                torch.int64
# layer1.2.bn3.weight  torch.Size([256])              torch.float32
# layer1.2.bn3.bias    torch.Size([256])              torch.float32
# layer2.0.conv1.weight torch.Size([15804])            torch.uint8
# layer2.0.conv1.codebook torch.Size([33])               torch.float64
# layer2.0.conv1.hftree torch.Size([137])              torch.uint8
# layer2.0.conv1.shape torch.Size([4])                torch.int64
# layer2.0.bn1.weight  torch.Size([128])              torch.float32
# layer2.0.bn1.bias    torch.Size([128])              torch.float32
# layer2.0.conv2.weight torch.Size([76978])            torch.uint8
# layer2.0.conv2.codebook torch.Size([33])               torch.float64
# layer2.0.conv2.hftree torch.Size([137])              torch.uint8
# layer2.0.conv2.shape torch.Size([4])                torch.int64
# layer2.0.bn2.weight  torch.Size([128])              torch.float32
# layer2.0.bn2.bias    torch.Size([128])              torch.float32
# layer2.0.conv3.weight torch.Size([29747])            torch.uint8
# layer2.0.conv3.codebook torch.Size([33])               torch.float64
# layer2.0.conv3.hftree torch.Size([137])              torch.uint8
# layer2.0.conv3.shape torch.Size([4])                torch.int64
# layer2.0.bn3.weight  torch.Size([512])              torch.float32
# layer2.0.bn3.bias    torch.Size([512])              torch.float32
# layer2.0.downsample.0.weight torch.Size([512, 256, 1, 1])   torch.float32
# layer2.0.downsample.1.weight torch.Size([512])              torch.float32
# layer2.0.downsample.1.bias torch.Size([512])              torch.float32
# layer2.1.conv1.weight torch.Size([29385])            torch.uint8
# layer2.1.conv1.codebook torch.Size([33])               torch.float64
# layer2.1.conv1.hftree torch.Size([137])              torch.uint8
# layer2.1.conv1.shape torch.Size([4])                torch.int64
# layer2.1.bn1.weight  torch.Size([128])              torch.float32
# layer2.1.bn1.bias    torch.Size([128])              torch.float32
# layer2.1.conv2.weight torch.Size([70477])            torch.uint8
# layer2.1.conv2.codebook torch.Size([33])               torch.float64
# layer2.1.conv2.hftree torch.Size([137])              torch.uint8
# layer2.1.conv2.shape torch.Size([4])                torch.int64
# layer2.1.bn2.weight  torch.Size([128])              torch.float32
# layer2.1.bn2.bias    torch.Size([128])              torch.float32
# layer2.1.conv3.weight torch.Size([29410])            torch.uint8
# layer2.1.conv3.codebook torch.Size([33])               torch.float64
# layer2.1.conv3.hftree torch.Size([137])              torch.uint8
# layer2.1.conv3.shape torch.Size([4])                torch.int64
# layer2.1.bn3.weight  torch.Size([512])              torch.float32
# layer2.1.bn3.bias    torch.Size([512])              torch.float32
# layer2.2.conv1.weight torch.Size([32906])            torch.uint8
# layer2.2.conv1.codebook torch.Size([33])               torch.float64
# layer2.2.conv1.hftree torch.Size([137])              torch.uint8
# layer2.2.conv1.shape torch.Size([4])                torch.int64
# layer2.2.bn1.weight  torch.Size([128])              torch.float32
# layer2.2.bn1.bias    torch.Size([128])              torch.float32
# layer2.2.conv2.weight torch.Size([75392])            torch.uint8
# layer2.2.conv2.codebook torch.Size([33])               torch.float64
# layer2.2.conv2.hftree torch.Size([137])              torch.uint8
# layer2.2.conv2.shape torch.Size([4])                torch.int64
# layer2.2.bn2.weight  torch.Size([128])              torch.float32
# layer2.2.bn2.bias    torch.Size([128])              torch.float32
# layer2.2.conv3.weight torch.Size([31957])            torch.uint8
# layer2.2.conv3.codebook torch.Size([33])               torch.float64
# layer2.2.conv3.hftree torch.Size([137])              torch.uint8
# layer2.2.conv3.shape torch.Size([4])                torch.int64
# layer2.2.bn3.weight  torch.Size([512])              torch.float32
# layer2.2.bn3.bias    torch.Size([512])              torch.float32
# layer2.3.conv1.weight torch.Size([34013])            torch.uint8
# layer2.3.conv1.codebook torch.Size([33])               torch.float64
# layer2.3.conv1.hftree torch.Size([137])              torch.uint8
# layer2.3.conv1.shape torch.Size([4])                torch.int64
# layer2.3.bn1.weight  torch.Size([128])              torch.float32
# layer2.3.bn1.bias    torch.Size([128])              torch.float32
# layer2.3.conv2.weight torch.Size([79799])            torch.uint8
# layer2.3.conv2.codebook torch.Size([33])               torch.float64
# layer2.3.conv2.hftree torch.Size([137])              torch.uint8
# layer2.3.conv2.shape torch.Size([4])                torch.int64
# layer2.3.bn2.weight  torch.Size([128])              torch.float32
# layer2.3.bn2.bias    torch.Size([128])              torch.float32
# layer2.3.conv3.weight torch.Size([32024])            torch.uint8
# layer2.3.conv3.codebook torch.Size([33])               torch.float64
# layer2.3.conv3.hftree torch.Size([137])              torch.uint8
# layer2.3.conv3.shape torch.Size([4])                torch.int64
# layer2.3.bn3.weight  torch.Size([512])              torch.float32
# layer2.3.bn3.bias    torch.Size([512])              torch.float32
# layer3.0.conv1.weight torch.Size([68101])            torch.uint8
# layer3.0.conv1.codebook torch.Size([33])               torch.float64
# layer3.0.conv1.hftree torch.Size([137])              torch.uint8
# layer3.0.conv1.shape torch.Size([4])                torch.int64
# layer3.0.bn1.weight  torch.Size([256])              torch.float32
# layer3.0.bn1.bias    torch.Size([256])              torch.float32
# layer3.0.conv2.weight torch.Size([317708])           torch.uint8
# layer3.0.conv2.codebook torch.Size([33])               torch.float64
# layer3.0.conv2.hftree torch.Size([137])              torch.uint8
# layer3.0.conv2.shape torch.Size([4])                torch.int64
# layer3.0.bn2.weight  torch.Size([256])              torch.float32
# layer3.0.bn2.bias    torch.Size([256])              torch.float32
# layer3.0.conv3.weight torch.Size([132586])           torch.uint8
# layer3.0.conv3.codebook torch.Size([33])               torch.float64
# layer3.0.conv3.hftree torch.Size([137])              torch.uint8
# layer3.0.conv3.shape torch.Size([4])                torch.int64
# layer3.0.bn3.weight  torch.Size([1024])             torch.float32
# layer3.0.bn3.bias    torch.Size([1024])             torch.float32
# layer3.0.downsample.0.weight torch.Size([1024, 512, 1, 1])  torch.float32
# layer3.0.downsample.1.weight torch.Size([1024])             torch.float32
# layer3.0.downsample.1.bias torch.Size([1024])             torch.float32
# layer3.1.conv1.weight torch.Size([128125])           torch.uint8
# layer3.1.conv1.codebook torch.Size([33])               torch.float64
# layer3.1.conv1.hftree torch.Size([137])              torch.uint8
# layer3.1.conv1.shape torch.Size([4])                torch.int64
# layer3.1.bn1.weight  torch.Size([256])              torch.float32
# layer3.1.bn1.bias    torch.Size([256])              torch.float32
# layer3.1.conv2.weight torch.Size([310815])           torch.uint8
# layer3.1.conv2.codebook torch.Size([33])               torch.float64
# layer3.1.conv2.hftree torch.Size([137])              torch.uint8
# layer3.1.conv2.shape torch.Size([4])                torch.int64
# layer3.1.bn2.weight  torch.Size([256])              torch.float32
# layer3.1.bn2.bias    torch.Size([256])              torch.float32
# layer3.1.conv3.weight torch.Size([126067])           torch.uint8
# layer3.1.conv3.codebook torch.Size([33])               torch.float64
# layer3.1.conv3.hftree torch.Size([137])              torch.uint8
# layer3.1.conv3.shape torch.Size([4])                torch.int64
# layer3.1.bn3.weight  torch.Size([1024])             torch.float32
# layer3.1.bn3.bias    torch.Size([1024])             torch.float32
# layer3.2.conv1.weight torch.Size([134599])           torch.uint8
# layer3.2.conv1.codebook torch.Size([33])               torch.float64
# layer3.2.conv1.hftree torch.Size([137])              torch.uint8
# layer3.2.conv1.shape torch.Size([4])                torch.int64
# layer3.2.bn1.weight  torch.Size([256])              torch.float32
# layer3.2.bn1.bias    torch.Size([256])              torch.float32
# layer3.2.conv2.weight torch.Size([315472])           torch.uint8
# layer3.2.conv2.codebook torch.Size([33])               torch.float64
# layer3.2.conv2.hftree torch.Size([137])              torch.uint8
# layer3.2.conv2.shape torch.Size([4])                torch.int64
# layer3.2.bn2.weight  torch.Size([256])              torch.float32
# layer3.2.bn2.bias    torch.Size([256])              torch.float32
# layer3.2.conv3.weight torch.Size([135905])           torch.uint8
# layer3.2.conv3.codebook torch.Size([33])               torch.float64
# layer3.2.conv3.hftree torch.Size([137])              torch.uint8
# layer3.2.conv3.shape torch.Size([4])                torch.int64
# layer3.2.bn3.weight  torch.Size([1024])             torch.float32
# layer3.2.bn3.bias    torch.Size([1024])             torch.float32
# layer3.3.conv1.weight torch.Size([138377])           torch.uint8
# layer3.3.conv1.codebook torch.Size([33])               torch.float64
# layer3.3.conv1.hftree torch.Size([137])              torch.uint8
# layer3.3.conv1.shape torch.Size([4])                torch.int64
# layer3.3.bn1.weight  torch.Size([256])              torch.float32
# layer3.3.bn1.bias    torch.Size([256])              torch.float32
# layer3.3.conv2.weight torch.Size([318751])           torch.uint8
# layer3.3.conv2.codebook torch.Size([33])               torch.float64
# layer3.3.conv2.hftree torch.Size([137])              torch.uint8
# layer3.3.conv2.shape torch.Size([4])                torch.int64
# layer3.3.bn2.weight  torch.Size([256])              torch.float32
# layer3.3.bn2.bias    torch.Size([256])              torch.float32
# layer3.3.conv3.weight torch.Size([133744])           torch.uint8
# layer3.3.conv3.codebook torch.Size([33])               torch.float64
# layer3.3.conv3.hftree torch.Size([137])              torch.uint8
# layer3.3.conv3.shape torch.Size([4])                torch.int64
# layer3.3.bn3.weight  torch.Size([1024])             torch.float32
# layer3.3.bn3.bias    torch.Size([1024])             torch.float32
# layer3.4.conv1.weight torch.Size([135386])           torch.uint8
# layer3.4.conv1.codebook torch.Size([33])               torch.float64
# layer3.4.conv1.hftree torch.Size([137])              torch.uint8
# layer3.4.conv1.shape torch.Size([4])                torch.int64
# layer3.4.bn1.weight  torch.Size([256])              torch.float32
# layer3.4.bn1.bias    torch.Size([256])              torch.float32
# layer3.4.conv2.weight torch.Size([321795])           torch.uint8
# layer3.4.conv2.codebook torch.Size([33])               torch.float64
# layer3.4.conv2.hftree torch.Size([137])              torch.uint8
# layer3.4.conv2.shape torch.Size([4])                torch.int64
# layer3.4.bn2.weight  torch.Size([256])              torch.float32
# layer3.4.bn2.bias    torch.Size([256])              torch.float32
# layer3.4.conv3.weight torch.Size([136579])           torch.uint8
# layer3.4.conv3.codebook torch.Size([33])               torch.float64
# layer3.4.conv3.hftree torch.Size([137])              torch.uint8
# layer3.4.conv3.shape torch.Size([4])                torch.int64
# layer3.4.bn3.weight  torch.Size([1024])             torch.float32
# layer3.4.bn3.bias    torch.Size([1024])             torch.float32
# layer3.5.conv1.weight torch.Size([138785])           torch.uint8
# layer3.5.conv1.codebook torch.Size([33])               torch.float64
# layer3.5.conv1.hftree torch.Size([137])              torch.uint8
# layer3.5.conv1.shape torch.Size([4])                torch.int64
# layer3.5.bn1.weight  torch.Size([256])              torch.float32
# layer3.5.bn1.bias    torch.Size([256])              torch.float32
# layer3.5.conv2.weight torch.Size([315990])           torch.uint8
# layer3.5.conv2.codebook torch.Size([33])               torch.float64
# layer3.5.conv2.hftree torch.Size([137])              torch.uint8
# layer3.5.conv2.shape torch.Size([4])                torch.int64
# layer3.5.bn2.weight  torch.Size([256])              torch.float32
# layer3.5.bn2.bias    torch.Size([256])              torch.float32
# layer3.5.conv3.weight torch.Size([134501])           torch.uint8
# layer3.5.conv3.codebook torch.Size([33])               torch.float64
# layer3.5.conv3.hftree torch.Size([137])              torch.uint8
# layer3.5.conv3.shape torch.Size([4])                torch.int64
# layer3.5.bn3.weight  torch.Size([1024])             torch.float32
# layer3.5.bn3.bias    torch.Size([1024])             torch.float32
# layer4.0.conv1.weight torch.Size([277600])           torch.uint8
# layer4.0.conv1.codebook torch.Size([33])               torch.float64
# layer4.0.conv1.hftree torch.Size([137])              torch.uint8
# layer4.0.conv1.shape torch.Size([4])                torch.int64
# layer4.0.bn1.weight  torch.Size([512])              torch.float32
# layer4.0.bn1.bias    torch.Size([512])              torch.float32
# layer4.0.conv2.weight torch.Size([1266589])          torch.uint8
# layer4.0.conv2.codebook torch.Size([33])               torch.float64
# layer4.0.conv2.hftree torch.Size([137])              torch.uint8
# layer4.0.conv2.shape torch.Size([4])                torch.int64
# layer4.0.bn2.weight  torch.Size([512])              torch.float32
# layer4.0.bn2.bias    torch.Size([512])              torch.float32
# layer4.0.conv3.weight torch.Size([552248])           torch.uint8
# layer4.0.conv3.codebook torch.Size([33])               torch.float64
# layer4.0.conv3.hftree torch.Size([137])              torch.uint8
# layer4.0.conv3.shape torch.Size([4])                torch.int64
# layer4.0.bn3.weight  torch.Size([2048])             torch.float32
# layer4.0.bn3.bias    torch.Size([2048])             torch.float32
# layer4.0.downsample.0.weight torch.Size([2048, 1024, 1, 1]) torch.float32
# layer4.0.downsample.1.weight torch.Size([2048])             torch.float32
# layer4.0.downsample.1.bias torch.Size([2048])             torch.float32
# layer4.1.conv1.weight torch.Size([541296])           torch.uint8
# layer4.1.conv1.codebook torch.Size([33])               torch.float64
# layer4.1.conv1.hftree torch.Size([137])              torch.uint8
# layer4.1.conv1.shape torch.Size([4])                torch.int64
# layer4.1.bn1.weight  torch.Size([512])              torch.float32
# layer4.1.bn1.bias    torch.Size([512])              torch.float32
# layer4.1.conv2.weight torch.Size([1299492])          torch.uint8
# layer4.1.conv2.codebook torch.Size([33])               torch.float64
# layer4.1.conv2.hftree torch.Size([137])              torch.uint8
# layer4.1.conv2.shape torch.Size([4])                torch.int64
# layer4.1.bn2.weight  torch.Size([512])              torch.float32
# layer4.1.bn2.bias    torch.Size([512])              torch.float32
# layer4.1.conv3.weight torch.Size([570730])           torch.uint8
# layer4.1.conv3.codebook torch.Size([33])               torch.float64
# layer4.1.conv3.hftree torch.Size([137])              torch.uint8
# layer4.1.conv3.shape torch.Size([4])                torch.int64
# layer4.1.bn3.weight  torch.Size([2048])             torch.float32
# layer4.1.bn3.bias    torch.Size([2048])             torch.float32
# layer4.2.conv1.weight torch.Size([567783])           torch.uint8
# layer4.2.conv1.codebook torch.Size([33])               torch.float64
# layer4.2.conv1.hftree torch.Size([137])              torch.uint8
# layer4.2.conv1.shape torch.Size([4])                torch.int64
# layer4.2.bn1.weight  torch.Size([512])              torch.float32
# layer4.2.bn1.bias    torch.Size([512])              torch.float32
# layer4.2.conv2.weight torch.Size([1307845])          torch.uint8
# layer4.2.conv2.codebook torch.Size([33])               torch.float64
# layer4.2.conv2.hftree torch.Size([137])              torch.uint8
# layer4.2.conv2.shape torch.Size([4])                torch.int64
# layer4.2.bn2.weight  torch.Size([512])              torch.float32
# layer4.2.bn2.bias    torch.Size([512])              torch.float32
# layer4.2.conv3.weight torch.Size([548465])           torch.uint8
# layer4.2.conv3.codebook torch.Size([33])               torch.float64
# layer4.2.conv3.hftree torch.Size([137])              torch.uint8
# layer4.2.conv3.shape torch.Size([4])                torch.int64
# layer4.2.bn3.weight  torch.Size([2048])             torch.float32
# layer4.2.bn3.bias    torch.Size([2048])             torch.float32
# fc.codebook          torch.Size([3])                torch.float64
# fc.bias              torch.Size([100])              torch.float32
# fc.weight            torch.Size([25601])            torch.uint8
# fc.hftree            torch.Size([10])               torch.uint8
