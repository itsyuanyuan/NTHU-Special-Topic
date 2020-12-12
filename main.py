import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchsummary import summary
import argparse
import os
import torch.utils.data
import torchvision.datasets as dset
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from model import myCNNmodel
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default = 'folder',help='folder')
parser.add_argument('--dataroot',default = "C:\\Users\\aspy1\\Desktop\\img\\origin",help='path to dataset')
parser.add_argument('--mdataroot',default = "C:\\Users\\aspy1\\Desktop\\img\\modify",help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=96, help='the height / width of the input image to network')
parser.add_argument('--step', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00015, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net(to continue training)")
parser.add_argument('--outf', default='./result/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default = 87,type=int, help='manual seed')
parser.add_argument('--BEC' , default = 0.5 , type = float , help = 'factor of BCELOSS')
parser.add_argument('--MSE' , default = 0.5 , type = float , help = 'factor of MSELOSS')
opt = parser.parse_args()
print(opt)

class mydataset(Dataset):
    def __init__(self,csv_path,img_path,transforms = None):
        self.data_info = pd.read_csv(csv_path,header = None)
        self.img_path = img_path
        self.transform = transforms
        self.X_train = np.asarray(self.data_info.iloc[:, 1])
        self.y_train = np.asarray(self.data_info.iloc[:, 0])

    def __getitem__(self, index):
        image_name = ''
        image_name = self.X_train[index][0] + image_name
        image_name = self.img_path + image_name
        img = Image.open(image_name)
        img_tensor = img
        if self.transform is not None:
            img_tensor = self.transform(img)
        label = self.y_train[index]
        return (img_tensor, label)
    def __len__(self):
        return len(self.data_info.index)

if __name__ == "__main__":
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    dataset_origin = dset.ImageFolder(root=opt.dataroot,
                               transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) )])
                               )
    
    dataset_modify = dset.ImageFolder(root = opt.mdataroot,
                                transform= transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])
                                )

    dataloader_x = torch.utils.data.DataLoader(dataset_origin, batch_size = opt.batchSize,
                                            shuffle = False, num_workers = int(opt.workers))

    dataloader_y = torch.utils.data.DataLoader(dataset_modify, batch_size = opt.batchSize,
                                            shuffle = False, num_workers = int(opt.workers))

    model = myCNNmodel(3)
    criterion_loglikelihood = nn.BCELoss()
    criterion_MSE = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    real_label = 1.0
    fake_label = 0

    for epoch in range (1,1+opt.step):
        for (i , data_x),(j,data_y) in zip(enumerate(dataloader_x,0),enumerate(dataloader_y,0)):
            model.zero_grad()
            X_cpu = data_x[0].to(device)
            batch_size = X_cpu.size(0)
            output = model(X_cpu)
            Y_cpu = data_y[0].to(device)
            loss = criterion_MSE(output , Y_cpu)
            loss.backward()
            print(loss)
            optimizer.step()
