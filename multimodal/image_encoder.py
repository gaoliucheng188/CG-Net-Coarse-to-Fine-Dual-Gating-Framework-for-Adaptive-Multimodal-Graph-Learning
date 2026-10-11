import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128, gpu=True):
        super().__init__()
        # Use torchvision weights enum to avoid deprecated pretrained flag
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 移除最后一层全连接
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(2048, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.gpu = gpu
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Do not call self.cuda() here; caller should move model to desired device.

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def encode_image(self, image_path):
        self.eval()
        with torch.no_grad():
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img).unsqueeze(0)
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            img = img.to(device)
            return self.forward(img)
            