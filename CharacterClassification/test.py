import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

# Transforms to be applied to Train-Test-Validation
test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(means, stds)])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net().to(device)

model.load_state_dict(torch.load('./CharacterClassification/weight.pth'))
model.eval()


def classify(image):
    img = Image.open(image)
    img = img.resize((32, 32), resample=0, box=None)
    image_tensor = test_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    # print(input.shape)
    output = model(input)
    index = output.data.cpu().numpy()
    # print(index)
    char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
    print(char[index.argmax()])
