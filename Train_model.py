import torch
import torch.nn.modules
import torch.utils
import torch.utils.data
import torchvision
import os
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
##################################################################################### SET up data ##########################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tranfroms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64)),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])

root_path = os.path.join(r"D:\machine_learning_AI_Builders\บท4\GAN\data")

data_set = torchvision.datasets.ImageFolder(root_path,transform=tranfroms)

data_loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=64,num_workers=0,shuffle=True)


############################################################################ สร้าง function ในการปรับเวทเริ่มต้น ###############################################################################

# ในเปเปอร์ DCGAN ทำการ generate pramiter ที่น้อยกว่าปกติ
# เราจะใช้ function นี้ในการปรับเวทเริ่มต้น ที่เราสร้างขึ้นมาจาก Generator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02) #mean = 0 ,std = 0.02
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data,1.,0.02)
        torch.nn.init.constant_(m.bias.data,0)

#################################################################################### Create Generator ####################################################################################

class Generator(torch.nn.Module):
    def __init__(self, nz: int, ngf : int, nc : int) -> None:   # ngf = จำนวนของ feature maps (หรือ channels) ใน Generator ที่ใช้ในแต่ละขั้นตอนของการทำงานเพื่อสร้างภาพ.
                                                                # nc  = num of chanel init image = 3 
                                                                # nz = lenght of latent vector
        super(Generator,self).__init__()
        self.main = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(num_features=ngf*8),
            torch.nn.ReLU( inplace=True), #inplace=True หมายถึงการดำเนินการแบบ "in-place" ซึ่งก็คือการเปลี่ยนแปลงค่าของเทนเซอร์เดิมโดยไม่สร้างเทนเซอร์ใหม่ขึ้นมา จะช่วยลดการใช้หน่วยความจำลงเนื่องจากไม่ต้องสร้างเทนเซอร์ใหม่

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
        return (self.main(input))
    
generator = Generator(nc=3,ngf=64,nz=100)
generator.apply(weights_init)
generator.to(device=device)


#################################################################################### Create Discriminator ####################################################################################

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
    
discriminator = Discriminator(nc=3,nfd=64)
discriminator.apply(weights_init)
discriminator.to(device=device)

#################################################################################### SET UP paramiter ####################################################################################

criterion  = torch.nn.BCELoss()  #น DCGAN (Deep Convolutional Generative Adversarial Network), criterion หมายถึงฟังก์ชันที่ใช้ในการคำนวณค่าขาดทุน (loss) ที่ใช้ในการฝึกโมเดล GAN โดยทั่วไปจะใช้แบบ Binary Cross Entropy Loss (BCE Loss)
fix_noise = torch.randn(64,100,1,1,device=device)
real_label = float(1)
fake_label = float(0)
lr = 3e-3
optimizerD = torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(0.5,0.999))
optimizerG = torch.optim.Adam(generator.parameters(),lr=lr,betas=(0.5,0.999))#Bata1 = 0.5

img_list = [] 
G_losses = []
D_losses = []
iters= 0

# pre csv logger

# columns = ["epoch","train_loss","valid_loss","accuracy","f1_score"]
# csv_df = pd.DataFrame(columns=columns)
# csv_file_name = "result_log.csv" 
# log_csv_path = os.path.join(os.path.join(r"D:\machine_learning_AI_Builders\บท4\GAN",csv_file_name)
num_epochs = 5
####################################################################################  Train model ####################################################################################
for epoch in range(num_epochs):
    for i,data in tqdm(enumerate(data_loader,0)):

        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################################################

        discriminator.zero_grad()
        real_img = data[0].to(device) #นำรูปในbatchมาประมวลผลใน GPU
        batch_size = real_img.shape[0]
        label = torch.full(size=(batch_size,),fill_value=real_label,dtype=torch.float,device=device)#ที่ใช้ในบรรทัดนั้นใช้สำหรับสร้าง Tensor ที่มีขนาดเท่ากับ batch_size และมีค่าทุกตัวเป็น real_label โดยใช้ device เป็น device ที่คุณกำหนด เช่น GPU (cuda:0) หรือ CPU (cpu)
        output = discriminator(real_img).view(-1) #การใช้ .view(-1) ในบรรทัดของโค้ดนั้นหมายถึงการเปลี่ยนรูปแบบของ Tensor โดยที่ -1 หมายถึงให้ PyTorch ปรับขนาดของ Tensor ให้เหมาะสมโดยอัตโนมัติ
        errD_real = criterion(output,label)
        errD_real.backward()
        D_x = output.mean().item()

        #สร้าง batch ของ latent vectors

        noise = torch.randn(batch_size,100,1,1,device=device)
        fack = generator(noise)
        label.fill_(fake_label)# fill_ เป็นเมทอดของ PyTorch Tensor ที่ใช้ในการกำหนดค่าใน Tensor ทั้งหมดให้มีค่าเดียวกันตามที่ระบุใน fake_label. ดังนั้น Tensor label จะมีค่าเป็น fake_label ////เป็นการกำหนดค่าใหม่ใน Tensor label โดยจะไม่ได้ทับค่า label ตัวเดิมที่มีอยู่แล้วแต่จะแทนที่ด้วย fake_label
        output = discriminator(fack.detach()).view(-1) # detach() ทำหน้าที่ให้ไม่มีการคำนวณ gradient ย้อนกลับไปยัง Generator ซึ่งแสดงให้เห็นว่า Generator จะไม่ถูกฝึกด้วยค่า gradient ที่ได้จากขั้นตอนนี้ของ Discriminator
        errD_fack = criterion(output,label)
        errD_fack.backward()
        errD = errD_fack+errD_real
        D_vs_G_d = output.mean().item()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fack).view(-1)
        errG = criterion(output,label)
        errG.backward()
        D_vs_G_g = output.mean().item()
        optimizerG.step()

        
        G_losses.append(errG.item())
        D_losses.append(errD.item())   

        if i % 50 == 0:
            print(f"{i}/{len(data_loader)} Loss_D{errD.item():.4f} Loss_G{errG.item():.4f} D(x):{D_x} D(G(z)):{D_vs_G_d}/{D_vs_G_g}")

        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(data_loader)-1)):
            with torch.no_grad():
                fake = generator(fix_noise).detach().cpu() #generator(fix_noise): เครือข่าย Generator ใช้เวกเตอร์สุ่มคงที่ (fix_noise) เพื่อสร้างภาพปลอม // .detach ป้องกันการคำนวณ Gradient สำหรับภาพเหล่านี้
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1

        

    print(f"{epoch}/{num_epochs} Loss_D{errD.item():.4f} Loss_G{errG.item():.4f} D(x):{D_x} D(G(z)):{D_vs_G_d}/{D_vs_G_g}")

####################################################################################  Save model ####################################################################################
        
model_path1 = os.path.join(os.path.join(r"D:\machine_learning_AI_Builders\บท4\GAN","modelG.pth"))
model_path2 = os.path.join(os.path.join(r"D:\machine_learning_AI_Builders\บท4\GAN","modelD.pth"))
torch.save(generator.state_dict(),model_path1)
torch.save(discriminator.state_dict(),model_path2)
print(f"***** Save Complete ******")


#################################################################################### Loss  ####################################################################################


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
# img = img * 0.5 + 0.5  # Unnormalize เพื่อให้อยู่ในช่วงค่า [0, 1]

# # แปลง img Tensor เป็น NumPy array และทำการ transpose เพื่อให้เป็นรูปแบบ (height, width, channels) สำหรับ matplotlib
# img_np = img.numpy()#.transpose((0, 2, 3, 1))  # (batch_size, height, width, channels)

# # # แสดงภาพเป็นเส้นกริดด้วย matplotlib
# # plt.figure(figsize=(10, 10))  # สร้าง figure ขนาด 10x10 นิ้ว
# # plt.axis("off")  # ปิดแสดงแกนที่มีใน figure
# # plt.title("Sample Images")  # ตั้งชื่อของ figure เป็น "Sample Images"
# # plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))  # แสดงภาพที่ได้จาก vutils.make_grid โดยทำการ transpose อีกครั้งเพื่อแสดงผล
# # plt.show()  # แสดงผลภาพ


#################################################################################### Gen image ####################################################################################
# Grab a batch of real images from the dataloader
real_batch = next(iter(data_loader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))#(1, 2, 0) ระบุการเรียงลำดับแกนใหม่ #การใช้ np.transpose(img_list[-1], (1, 2, 0)) เปลี่ยนแกนจาก (C, H, W) ไปเป็น (H, W, C) เพื่อให้เหมาะสมกับการแสดงผล
plt.show()