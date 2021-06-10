import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

# Transforms to be applied to Train-Test-Validation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)])


class CustomImageDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

        self.data = []
        self.label = []
        for i in os.listdir(self.img_path):
            if os.path.isdir(self.img_path + "/" + i):
                for j in os.listdir(self.img_path + "/" + i):
                    self.data.append(self.img_path + "/" + i + "/" + j)
                    self.label.append(str(i))
        label_encoder = preprocessing.LabelEncoder()
        self.label = label_encoder.fit_transform(self.label)
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(mapping)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = train_transforms(img)
        label = self.label[idx]
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        return {'im': img, 'labels': label_tensor}


dataset = CustomImageDataset("./doc")
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)


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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    print("EPOCH:", epoch, end=" ")
    running_loss = 0
    running_acc = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data['im'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        result = torch.argmax(outputs, dim=1)
        running_loss += loss.item()
        running_acc += torch.mean((result == labels).type(torch.float))

        loss.backward()
        optimizer.step()

    else:
        train_loss = running_loss / len(dataloader)
        train_acc = running_acc / len(dataloader)

        print("Training Loss: {:.3f}".format(train_loss), end=" ")

        print("Train Accuracy: {:.2f}%".format(train_acc.item() * 100))

torch.save(model.state_dict(), "weight.pth")
