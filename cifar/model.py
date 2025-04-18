import torch
import torchvision
from torch import nn
import timm
from torchvision import transforms

def create_model(num_classes:int=10,
                 seeds:int=42):
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
  
  model = timm.create_model('vit_tiny_patch16_224',pretrained=True)
  model.head = nn.Linear(in_features=model.head.in_features,out_features=10)

  return model,transform
