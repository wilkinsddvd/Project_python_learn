import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the DigitRecognizer model with three hidden layers
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, optimizer, and learning rate scheduler
model = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training and validation loop
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'digit_recognizer.pth')
print('Model trained and saved as digit_recognizer.pth')

# 优化
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
#
# # Define the DigitRecognizer model
# class DigitRecognizer(nn.Module):
#     def __init__(self):
#         super(DigitRecognizer, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64*7*7)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # Hyperparameters
# batch_size = 64
# learning_rate = 0.001
# num_epochs = 10  # Increased number of epochs
#
# # Data augmentation and preprocessing
# transform = transforms.Compose([
#     transforms.RandomRotation(10),  # Random rotation
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # Load datasets
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
#
# # Initialize model, loss function, optimizer, and learning rate scheduler
# model = DigitRecognizer()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Learning rate scheduler
#
# # Training and validation loop
# for epoch in range(num_epochs):
#     model.train()
#     for i, (images, labels) in enumerate(train_loader):
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#
#     # Step the scheduler
#     scheduler.step()
#
#     # Validation
#     model.eval()
#     val_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             outputs = model(images)
#             val_loss += criterion(outputs, labels).item()
#             pred = outputs.argmax(dim=1, keepdim=True)
#             correct += pred.eq(labels.view_as(pred)).sum().item()
#
#     val_loss /= len(val_loader.dataset)
#     accuracy = 100. * correct / len(val_loader.dataset)
#     print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
#
# # Save the model
# torch.save(model.state_dict(), 'digit_recognizer.pth')
# print('Model trained and saved as digit_recognizer.pth')

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
#
# # 定义手写数字识别模型
# class DigitRecognizer(nn.Module):
#     def __init__(self):
#         super(DigitRecognizer, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*7*7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64*7*7)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # 超参数
# batch_size = 64
# learning_rate = 0.001
# num_epochs = 5
#
# # 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # 加载数据集
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# # 初始化模型、损失函数和优化器
# model = DigitRecognizer()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练模型
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#
# # 保存模型
# torch.save(model.state_dict(), 'digit_recognizer.pth')
# print('Model trained and saved as digit_recognizer.pth')