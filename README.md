# Light-Weight-Neural-Network-Design-and-Implementation-on-Low-power-Processors
Deep Compression Pipeline + BranchyNet

Main stages include: Branch Adding -> Pruning -> Quantization -> Coding

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

