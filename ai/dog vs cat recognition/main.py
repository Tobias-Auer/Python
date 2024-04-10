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

# Define transformations
transform = transforms.Compose([
    transforms.Resize(128),  # Decrease image resolution
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Define a function to load and preprocess images
def load_images(file_paths):
    images = []
    targets = []
    for f in file_paths:
        img = Image.open(f)
        img_tensor = transform(img)
        images.append(img_tensor)
        # Determine target (cat or dog)
        if "cat" in f:
            targets.append(0)
        elif "dog" in f:
            targets.append(1)
    return torch.stack(images), torch.tensor(targets)


# Define a function to create batches from file paths
def create_batches(file_paths, batch_size):
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        yield load_images(batch_files)


# Define file paths for training data
train_dir = r"C:\Users\Tobias\Downloads\dogs-vs-cats\train"
train_files = [os.path.join(train_dir, f) for f in listdir(train_dir)]


# Define model architecture
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=5)  # New convolutional layer
        self.fc1 = nn.Linear(207360, 1000)  # Adjust input size based on image resolution
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # New convolutional layer
        x = x.view(-1, 207360)  # Adjust size based on image resolution
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)



# Instantiate model
model = Netz()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Define training loop
def train(epoch, train_files, batch_size):
    model.train()
    for batch_id, (data, target) in enumerate(create_batches(train_files, batch_size)):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_id * batch_size, len(train_files),
                                                                       100. * batch_id / len(train_files), loss.item()))


# Training loop
batch_size = 60  # Decrease batch size to reduce memory usage
for epoch in range(1, 2):
    print(f"--------------- STARTING EPOCH {epoch} FROM 30 ---------------")
    train(epoch, train_files, batch_size)


def test():
    model.eval()
    files = listdir(r"C:\Users\Tobias\Downloads\dogs-vs-cats\test")
    f = random.choice(files)
    img = Image.open(fr"C:\Users\Tobias\Downloads\dogs-vs-cats\test\{f}")
    img_eval_tensor = transform(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim=True)[1])
    img.show()
    input()



while True:
    test()
