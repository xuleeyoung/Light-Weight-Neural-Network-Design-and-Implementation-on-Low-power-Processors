# Light-Weight-Neural-Network-Design-and-Implementation-on-Low-power-Processors
Deep Compression Pipeline + BranchyNet
Testing Procedures:
1. Download models from torchvision
```
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```
