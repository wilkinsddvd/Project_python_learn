import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 定义生成器
class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.gen(z)
        out = out.view(out.shape[0], 1, 28, 28)
        return out

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.dis(img.view(img.shape[0], -1))
        return out

# 超参数
z_dim = 100  # 噪声向量的维度
image_dim = 784  # 图像的维度 (28*28)
batch_size = 64
lr = 0.0002
epochs = 5

# 加载数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(z_dim, image_dim).to(device)
netD = Discriminator(image_dim).to(device)

# 优化器
optimizerG = optim.Adam(netG.parameters(), lr=lr)
optimizerD = optim.Adam(netD.parameters(), lr=lr)

# 损失函数
criterion = nn.BCELoss()

# 训练
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # 训练判别器
        optimizerD.zero_grad()
        real_outputs = netD(imgs.to(device))
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        fake_outputs = netD(netG(torch.randn(imgs.size(0), z_dim).to(device)).detach())
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()
        optimizerD.step()

        # 训练生成器
        optimizerG.zero_grad()
        outputs = netD(netG(torch.randn(imgs.size(0), z_dim).to(device)))
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'D Loss: {real_loss.item() + fake_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 保存生成的图像
    with torch.no_grad():
        fake_images = netG(torch.randn(batch_size, z_dim).to(device))
        save_image(fake_images, f'epoch_{epoch+1}.png', normalize=True)

print('Training finished.')