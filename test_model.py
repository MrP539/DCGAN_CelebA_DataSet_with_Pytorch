# Import libraries
import torch
import os
import torchvision.utils as vutils
# Display a generated image
import matplotlib.pyplot as plt
import numpy as np

# Define Generator and Discriminator class (ใช้โครงสร้างเดียวกับที่สร้างตอนแรก)
class Generator(torch.nn.Module):
    def __init__(self, nz: int, ngf : int, nc : int) -> None:
        super(Generator,self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=ngf*8),
            torch.nn.ReLU( inplace=True),
            torch.nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1 ,bias=False),
            torch.nn.BatchNorm2d(ngf*4),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=ngf*4,out_channels=ngf*2,kernel_size=4,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(num_features=ngf*2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=ngf*2, out_channels= ngf, kernel_size=4, stride=2 ,padding=1 , bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2, padding=1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self,input):
        return self.main(input)

class Discriminator(torch.nn.Module):
    def __init__(self,nc:int,nfd:int):
        super(Discriminator,self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=nc,out_channels=nfd,kernel_size=4,stride=2,padding=1,bias=False),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Conv2d(in_channels=nfd,out_channels=nfd*2,kernel_size=4,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(nfd*2),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Conv2d(in_channels=nfd*2,out_channels=nfd*4,kernel_size=4,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(nfd*4),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Conv2d(in_channels=nfd*4,out_channels=nfd*8,kernel_size=4,stride=2,padding=1,bias=False),
            torch.nn.BatchNorm2d(nfd*8),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Conv2d(in_channels=nfd*8,out_channels=1,kernel_size=4,stride=1,padding=0,bias=False),
            torch.nn.Sigmoid()
        )
    
    def forward(self,input):
        return self.main(input)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create model objects
generator = Generator(nc=3, ngf=64, nz=100).to(device)
discriminator = Discriminator(nc=3, nfd=64).to(device)

# Load the saved state_dict
model_path1 = os.path.join(r"D:\machine_learning_AI_Builders\บท4\GAN\result\model", "modelG.pth")
model_path2 = os.path.join(r"D:\machine_learning_AI_Builders\บท4\GAN\result\model", "modelD.pth")
generator.load_state_dict(torch.load(model_path1))
discriminator.load_state_dict(torch.load(model_path2))

# Set the models to evaluation mode
generator.eval()
discriminator.eval()

# Test the loaded models (optional)
# You can now use the generator to generate images
fixed_noise = torch.randn(64, 100, 1, 1, device=device)
with torch.no_grad():
    fake_images = generator(fixed_noise).detach().cpu()


# Convert the tensor to a numpy array and transpose the dimensions
image_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
image_np = np.transpose(image_grid, (1, 2, 0))

# Display the image
plt.title("Generate Images")
plt.imshow(image_np)
plt.show()
