import os
import random
import torch.optim as optim
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# Define the same transformations used during training
normalize = transforms.Normalize(
    mean=[
        0.485,
        0.456,
        0.406
    ],
    std=[
        0.229,
        0.224,
        0.225
    ]
)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize
])


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # ... (rest of your model architecture definition, ensuring it matches the trained model)

    def forward(self, x):
        # ... (forward pass implementation)
        return F.sigmoid(x)  # Assuming sigmoid output for cat/dog probabilities


def load_model(path):
    """Loads a saved model from a specified path"""
    model = Netz()  # Create a new instance of the model architecture
    model.load_state_dict(torch.load(path))
    return model


def test_image(model, path):
    """Tests a single image using the loaded model"""
    img = Image.open(path)
    img_tensor = transform(img)
    img_tensor.unsqueeze_(0)  # Add batch dimension
    data = img_tensor.cuda() if torch.cuda.is_available() else img_tensor
    out = model(data)
    prob = torch.softmax(out, dim=1)  # Calculate class probabilities
    cat_prob, dog_prob = prob.data.cpu().numpy()[0]
    print(f"Cat probability: {cat_prob:.4f}, Dog probability: {dog_prob:.4f}")
    if cat_prob > dog_prob:
        print("Predicted: Cat")
    else:
        print("Predicted: Dog")


# Assuming you've addressed model architecture changes and the saved model path is correct:
model_path = "model_epoch_35.pth"  # Replace with the actual path to your saved model
test_image_path = "path/to/your/test/image.jpg"  # Replace with the path to the test image

try:
    model = load_model(model_path)
    test_image(model, test_image_path)
except RuntimeError as e:
    if "Unexpected key(s)" in str(e):
        print("Error: Model architecture mismatch. Ensure the model definition matches the saved model.")
    else:
        print(f"An error occurred: {e}")
