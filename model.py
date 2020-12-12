model.py
Y
張
陳
汪
類型
文字
大小
6 KB (6,457 個位元組)
儲存空間使用量
0 個位元組擁有者：清華大學
位置
2018CNN
建立者
我
上次修改時間
我在2019年5月10日修改過
上次開啟時間
我在上午10:52開啟過
建立日期
2019年4月24日
新增說明
檢視者可以下載
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import argparse


class myCNNmodel(nn.Module):
    def __init__(self,C):
        super(myCNNmodel,self).__init__()
        self.C = C
        self.model = nn.Sequential(
            nn.Conv2d(C,C*4,kernel_size = 3 , stride= 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*4),
            nn.Conv2d(C*4,C*16,kernel_size = 1, stride= 2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*16),
            nn.Conv2d(C*16,C*64,kernel_size = 5, stride= 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*64),
            nn.Conv2d(C*64,C*64,kernel_size = 1, stride= 2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*64),
            nn.ConvTranspose2d(C*64,C*16,kernel_size = 2, stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*16),
            nn.ConvTranspose2d(C*16,C*4,kernel_size = 2 , stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*4),
            nn.ConvTranspose2d(C*4,C*4,kernel_size = 7, stride=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(C*4),
            nn.ConvTranspose2d(C*4,C,kernel_size = 3 , stride=1),
            nn.Tanh()
        )
    def forward(self,x):   
        output = self.model(x)
        return output     

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def double_conv_s2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2) 
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        x = self.dconv_down4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

def sample_from_noise(opt,x):
    if torch.cuda.is_available():
        return torch.randn(opt.batchSize,8,x.shape[2],x.shape[3]).cuda()
    else:
        return torch.randn(opt.batchSize,8,x.shape[2],x.shape[3])

class Unet_noise(nn.Module):
    def __init__(self,opt):
        super(Unet_noise,self).__init__()
        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample= nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        self.opt = opt
        self.dconv_up3 = double_conv(128+8+256, 128+4)
        self.dconv_up2 = double_conv(128+64+4, 64+2)
        self.dconv_up1 = double_conv(64+16+2, 16+1)
        self.conv_last = nn.Conv2d(16+1, 3, 1)
    
    def forward(self , x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2) 
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        x = self.dconv_down4(x)
        noise = sample_from_noise(opt,x)
        x = torch.cat([x,noise],dim=1)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

class discrem(nn.Module):
    def __init__(self):
        super(discrem,self).__init__()
        self.fc = nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()
        self.sq = nn.Sequential(
            double_conv_s2(3,16),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            double_conv_s2(16,64),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            double_conv_s2(64,128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            double_conv_s2(128,256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,2),
        )
    def forward(self,x):
        out = self.sq(x)
        out = out.view(-1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',default = True ,action='store_true', help='enables cuda')
    parser.add_argument('--lr', default = 0.0001 , type = float , help = 'learning rate')
    parser.add_argument('--step' , default = 1000 , type = int , help = 'training step')
    parser.add_argument('--batchSize' , default = 2 , type = int , help = 'batch size')

    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = Unet().to(device)
    #model = Unet_noise(opt).to(device)
    model = discrem().to(device)
    summary(model,(3,96,96))
