from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import timm
import time

import pathlib

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.ops import box_iou

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define first detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detector = YOLO("weights/detector.pt")

# Define classifier
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

Filter = models.vgg16(weights='DEFAULT')
for param in list(Filter.features.parameters())[:-4]:  # Keep last conv block trainable
        param.requires_grad = False
    
# Modify the classifier (fully connected layers)
# VGG16's classifier has 6 FC layers, we'll replace them all
num_features = Filter.classifier[0].in_features

# Create new classifier
Filter.classifier = nn.Sequential(
    nn.Linear(num_features, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 512),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

Filter.load_state_dict(torch.load("weights/filter.pth", map_location=device))
Filter.to(device)
Filter.eval()

# Define classes
classes = ['Disorganized_pillow', 'Messy Table', 'dirty_bathtub', 'dirty_floor', 'disorganized_towel', 'mess', 'messy_bed', 'messy_sink', 'messy_table']
# Load model
# Patch PosixPath to WindowsPath for loading checkpoints saved on Linux
pathlib.PosixPath = pathlib.WindowsPath
checkpoint = torch.load('./weights/classifier.pth', map_location=device)


class EfficientNetClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=10, pretrained=True, dropout=0.3):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'  # Global average pooling
        )
        
        # Get feature dimension
        num_features = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
def classify_image(image, model):
    image = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        outputs = model(image)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def forward_pass_single_class(model, image_tensor, device):
    """
    Perform forward pass on a single image or batch and return only the predicted class
    
    Args:
        model: Trained model
        image_tensor: Input tensor (C, H, W) or (N, C, H, W)
        device: Device to run inference on
    
    Returns:
        predicted_class_idx: Single predicted class index (int for single image, array for batch)
    """
    model.eval()
    
    with torch.no_grad():
        # Handle single image (add batch dimension)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            single_image = True
        else:
            single_image = False
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(image_tensor)
        else:
            logits = model(image_tensor)
        
        # Get predictions (highest probability class)
        _, predictions = torch.max(logits, 1)
        
        # Return single class index
        if single_image:
            return predictions.item()  # Return as single integer
        else:
            return predictions.cpu().numpy()  # Return as numpy array for batch

def predict_single_image_class(model, image, transform, classes, device):
    """
    Predict class for a single image from file path - returns only class name
    
    Args:
        model: Trained model
        image: Path to image file
        transform: Preprocessing transforms
        classes: List of class names
        device: Device to run inference on
    
    Returns:
        predicted_class_name: String with the predicted class name
    """
    # Load and preprocess image
    image = np.array(image.convert('RGB'))  # Ensure image is in RGB format
    
    # Apply transforms
    if transform:
        transformed = transform(image=image)
        image_tensor = transformed['image']
    else:
        # Basic transform if none provided
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    
    # Forward pass
    predicted_class_idx = forward_pass_single_class(model, image_tensor, device)
    
    # Return class name
    return classes[predicted_class_idx]

def get_valid_transforms(image_size=384):
    """Validation/test transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
])
    
# Create model
model = EfficientNetClassifier(
    model_name=checkpoint['config']['model_name'],
    num_classes=len(classes),
    pretrained=False
)
    
# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
transform = get_valid_transforms()

classes = ['Disorganized_pillow', 'Messy Table', 'dirty_bathtub', 'dirty_floor', 'disorganized_towel', 'mess', 'messy_bed', 'messy_sink', 'messy_table']
    
def Inference(img: Image.Image, service_to_mess: dict): 
    predictions = detector(img, conf=0.5)[0]
    
    boxes = predictions.boxes
    width, height = img.size
    
    mess_to_service = {}
    for Class in classes:
        mess_to_service[Class] = set()
    for service, messes in service_to_mess.items():
        for mess in messes:
            mess_to_service[mess].add(service)
    print(mess_to_service)
    messes, services = set(), set()
    
    for box in boxes:
        x, y, w, h = map(float, box.xywh[0].tolist())
        conf = float(box.conf[0])
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        confirmation = classify_image(img.crop((x1, y1, x2, y2)), Filter)
        if confirmation == 1:
            continue
        class_name = predict_single_image_class(model, img.crop((x1, y1, x2, y2)), transform, classes, device)
        messes.add(class_name)
        for service in mess_to_service[class_name]:
            services.add((service, conf / len(service_to_mess[service])))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        font_size = 18
        font=ImageFont.truetype("arial.ttf",font_size )
        draw.text((x1 + 1, y1), f"{class_name} {conf:.2f}", fill='red', font=font)
    return img, services

if __name__ == "__main__":
    # Example usage
    img_path = "./datanew/test/images/IMG_0613_png.rf.560fcc14458d84a5ba0a4703ab860ea7.jpg"
    img = Image.open(img_path)
    
    service_to_mess = {
        'cleaning': ['dirty_bathtub', 'dirty_floor'],
        'laundry': ['disorganized_towel', 'messy_bed'],
        'dishes': ['messy_sink', 'messy_table']
    }
    
    result_img, services = Inference(img, service_to_mess)
    result_img.show()  # Display the image with annotations
    print("Services needed:", services)