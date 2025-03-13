import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the DigitRecognizer model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model securely
model = DigitRecognizer()
model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cuda:0'), weights_only=True))
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Recognize digit in the image
def recognize_digit(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

if __name__ == "__main__":
    # Specify the absolute path to the image
    image_path = 'D:/test_image/1.png'
    digit = recognize_digit(image_path)
    print(f"Recognized Digit: {digit}")

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
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
# # Load the pre-trained model securely
# model = DigitRecognizer()
# # 使用`weights_only=True`来安全加载模型权重
# model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cuda:0'), weights_only=True))
# model.eval()
#
# # Preprocess the input image
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize((28, 28))  # Resize to 28x28
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image
#
# # Recognize digit in the image
# def recognize_digit(image_path):
#     image = preprocess_image(image_path)
#     with torch.no_grad():
#         output = model(image)
#         prediction = torch.argmax(output, dim=1).item()
#     return prediction
#
# if __name__ == "__main__":
#     # Specify the absolute path to the image
#     image_path = 'D:/test_image/4.png'
#     digit = recognize_digit(image_path)
#     print(f"Recognized Digit: {digit}")

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
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
# # Load the pre-trained model
# model = DigitRecognizer()
# model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cuda:0')))
# model.eval()
#
# # Preprocess the input image
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize((28, 28))  # Resize to 28x28
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image
#
# # Recognize digit in the image
# def recognize_digit(image_path):
#     image = preprocess_image(image_path)
#     with torch.no_grad():
#         output = model(image)
#         prediction = torch.argmax(output, dim=1).item()
#     return prediction
#
# if __name__ == "__main__":
#     # Specify the absolute path to the image
#     image_path = 'D:/test_image/9.png'
#     digit = recognize_digit(image_path)
#     print(f"Recognized Digit: {digit}")

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import sys
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
# # Load the pre-trained model
# model = DigitRecognizer()
# model.load_state_dict(torch.load('digit_recognizer.pth', map_location=torch.device('cuda:0')))
# model.eval()
#
# # Preprocess the input image
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize((28, 28))  # Resize to 28x28
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image
#
# # Recognize digit in the image
# def recognize_digit(image_path):
#     image = preprocess_image(image_path)
#     with torch.no_grad():
#         output = model(image)
#         prediction = torch.argmax(output, dim=1).item()
#     return prediction
#
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python recognize_digit.py <image_path>")
#         sys.exit(1)
#
#     image_path = sys.argv[1]
#     digit = recognize_digit(image_path)
#     print(f"Recognized Digit: {digit}")