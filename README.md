# Light-Weight-Neural-Network-Design-and-Implementation-on-Low-power-Processors
Deep Compression Pipeline + BranchyNet

Main stages include: Branch Adding -> Pruning -> Quantization -> Coding

### Experiment Results:

For VGG-16, the pipeline achieves storage reduction from 527.8 MB to 28 MB, and running time saving from 172504 ms to 74908 ms on CIFAR-10 test dataset (Hardware: NVIDIA Tesla K-80). 

|  VGG-16 | Original | Branch Adding | Pruning | Quantization | Coding |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Accuracy | 86.83% | 85.7% | 85.32% | 84.7% | 84.7% |
| Storage (MB) | 527.8 | 625.9 | 625.9 | 156.6 | 28 |
| Running time (ms) | 172504 | 73061 | 71855 | 75385 | 74908 |

For ResNet-50, the pipeline achieves storage reduction from 90.8 MB to 21.2 MB, and running time saving from 112699 ms to 103726 ms on CIFAR-100 test dataset (Hardware: NVIDIA GeForce RTX 2080). 

|  ResNet-50 | Original | Branch Adding | Pruning | Quantization | Coding |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Accuracy | 58.4% | 52.82% | 52.77% | 52.68% | 52.68% |
| Storage (MB) | 90.8 | 91 | 91 | 31.3 | 21.2 |
| Running time (ms) | 112699 | 107054 | 107912 | 107653 | 103726 |


### Compressing Procedures:
1. Download models from torchvision

Uncomment following lines in `branch_cnn.py`.
```
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

2. Add side branches to original network
```
python branch_cnn.py
```

3. Pruning
```
python prune.py
```

4. Quantization
```
python quant.py
python quant_test.py
```

5. Huffman Coding
```
python huffman_modules.py
python huffman.py
```

The resulting light-weight network parameter files and itermediate files are all saved in `/saves`.

### Testing Only (After Compressing)

### Reference
S. Han, H. Mao, and W. Dally, "Deep Compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding", in ICLR, 2016.

S. Teerapittayanon, B. McDanel and H. T. Kung, "BranchyNet: Fast inference via early exiting from deep neural networks," 2016 23rd International Conference on Pattern Recognition (ICPR), Cancun, Mexico, 2016.
