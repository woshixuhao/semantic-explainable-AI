import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter
from network import feature_net
from collections import OrderedDict
import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import cv2
import numpy as np
import argparse
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import cv2
import numpy as np
import torch
import random
import os
from torch.autograd import Variable
from torch.nn import Linear,Tanh,Sequential
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import heapq
from scipy import interpolate

np.set_printoptions(threshold=10000)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_seed', type=int, default=324, help='random seeds for torch')
    parser.add_argument('--population', type=int, default=50, help='population siza')
    parser.add_argument('--n_segment', type=int, default=40, help='length of genome')
    parser.add_argument('--prob_mut', type=float, default=0.5, help='mutation prob')
    parser.add_argument('--animal', type=str, default='dog', help='cat or dog')
    parser.add_argument('--iter_num', type=int, default=50, help='iter num')
    parser.add_argument('--pic_num', type=int, default=500, help='picture_num')
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config
config=parse_args()
torch.manual_seed(config.torch_seed)
path = r'D:\pycharm project\VAE_PDE\dogs_vs_cats/'
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),
                                      transform=transform)
              for x in ["train", "val"]}

data_image_dog = datasets.ImageFolder(root=os.path.join(path, 'train_dog'),
                                      transform=transform)
data_image_cat = datasets.ImageFolder(root=os.path.join(path, 'train_cat'),
                                      transform=transform)

data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=1,
                                                    shuffle=True)
                     for x in ["train", "val"]}
data_loader_dog=torch.utils.data.DataLoader(dataset=data_image_dog,
                                                    batch_size=1,
                                                    shuffle=True)
data_loader_cat=torch.utils.data.DataLoader(dataset=data_image_cat,
                                                    batch_size=1,
                                                    shuffle=True)


parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--pre_epoch', default=0, help='begin epoch')
parser.add_argument('--total_epoch', default=1, help='time for ergodic')
parser.add_argument('--model', default='vgg', help='model for training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--pre_model', default=False, help='use pre-model')  # 恢复训练时的模型路径
args = parser.parse_args()

# 定义使用模型
model = args.model
# 使用gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
use_model = feature_net(model, dim=512, n_classes=2) #for vgg dim=512
use_model.load_state_dict(torch.load("CNN_model_save/VGG_224.pth"))
use_model.eval() #冻结参数，以防相同输入，不同输出

for parma in use_model.feature.parameters():
    parma.requires_grad = False

for index, parma in enumerate(use_model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

use_model = use_model.to(device)

class lime_GA():
    def __init__(self,config):
        self.prob_mut=config.prob_mut
        self.population=config.population
        self.length=config.n_segment
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.iter_num=config.iter_num
        self.sigma=5
        self.picture_num=config.pic_num
        self.picture_shape=224
        self.animal=config.animal


    def Random_Genome(self):
        random_genome=np.random.randint(0,2,(self.population,self.length))
        return random_genome

    def Translate_Genome(self,segments,image_origin):
        genome=self.Chrom
        # N是samples的个数
        mask = np.zeros([self.population,image_origin.shape[0],image_origin.shape[1]], dtype="uint8") #[224,224]
        for i in range(self.population):
            for j in np.where(genome[i]==1)[0]:
                mask[i,segments == j] = 255
        return mask

    def mutation(self):
        # mutation of 0/1 type chromosome
        mask = (np.random.rand(self.population,self.length) < self.prob_mut) * 1
        self.Chrom ^= mask
        return self.Chrom

    def crossover(self):
        Chrom, size_pop, len_chrom = self.Chrom, self.population, self.length
        np.random.shuffle(Chrom)
        Chrom1, Chrom2 = Chrom[::2], Chrom[1::2]
        mask = np.zeros(shape=(int(size_pop / 2), len_chrom), dtype=int)
        for i in range(int(size_pop / 2)):
            n1, n2 = np.random.randint(0, self.length, 2)
            if n1 > n2:
                n1, n2 = n2, n1
            mask[i, n1:n2] = 1
        mask2 = (Chrom1 ^ Chrom2) & mask
        Chrom1 ^= mask2
        Chrom2 ^= mask2
        Chrom[::2], Chrom[1::2] = Chrom1, Chrom2
        self.Chrom = Chrom
        return self.Chrom

    def recover_to_image(self,image_origin):
        img = image_origin.copy()
        img = (img - self.mean) / self.std
        img = img.transpose((2, 0, 1))
        image = torch.from_numpy(img.astype(np.float32)).reshape([1, 3, 224, 224])
        return image

    def Calculate_Fitness(self,mask, img):
        image_total=torch.zeros([self.population,img.shape[2],img.shape[0],img.shape[1]])
        for i in range(self.population):
            img_split = np.multiply(img, cv2.cvtColor(mask[i], cv2.COLOR_GRAY2BGR) > 0)
            image_total[i] = lime_GA.recover_to_image(self,img_split)
        image_total= Variable(image_total.to(config.device))

        label_prediction = use_model(image_total)[0]
        if self.animal=='dog':
            error = (label_prediction[:, 1] - label_prediction[:, 0]).cpu().data.numpy()
        elif self.animal=='cat':
            error = (label_prediction[:, 0] - label_prediction[:, 1]).cpu().data.numpy()
        return error

    def Select(self,error): # nature selection wrt pop's fitness
        select_Chrom=self.Chrom.copy()
        select_error=error.copy()
        re1 = list(map(list(error).index, heapq.nlargest(int(self.population/2), error)))
        num=0
        for index in re1:
            select_Chrom[num]=self.Chrom[index]
            select_error[num]=error[index]
            num+=1
        select_Chrom[int(self.population/2):]=np.random.randint(0,2,(self.population-int(self.population/2),self.length))
        self.Chrom=select_Chrom
        return select_error

    def GA(self):
        image_best_save=np.zeros([self.picture_num,3,224,224])
        image_origin_save=np.zeros([self.picture_num,3,224,224])
        if self.animal=='dog':
            dataset=data_loader_dog
        if self.animal=='cat':
            dataset=data_loader_cat
        for epoch, data in enumerate(dataset):
            print('------------第%d张图处理----------------' % (epoch))
            image, label = data

            img = torchvision.utils.make_grid(image)
            img = img.cpu().data.numpy().transpose((1, 2, 0))  # 本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
            img = img * self.std + self.mean  # (228, 906, 3)范围由(-1, 1)变成(0, 1)
            # plt.imshow(img)
            # plt.show()

            # 查看分割图
            segments = slic(img_as_float(img), n_segments=self.length, sigma=self.sigma)
            self.Chrom=lime_GA.Random_Genome(self)
            for iter in range(self.iter_num):
                mask=lime_GA.Translate_Genome(self,segments,img)
                error=lime_GA.Calculate_Fitness(self,mask,img)
                select_error=lime_GA.Select(self,error)
                best=self.Chrom.copy()[0]
                self.Chrom = lime_GA.crossover(self)
                self.Chrom=lime_GA.mutation(self)
                self.Chrom[0]=best
                print("iter: %d, best fiteness: %s" % (iter, select_error[0]))
            mask = np.zeros([img.shape[0], img.shape[1]], dtype="uint8")  # [224,224]
            for j in np.where(best == 1)[0]:
                mask[segments == j] = 255
            img_split = np.multiply(img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0)
            image_best=lime_GA.recover_to_image(self, img_split)
            image_best_save[epoch]=image_best.data.numpy()
            image_origin_save[epoch]=image.cpu().data.numpy()
            # plt.imshow(img_split)
            # plt.show()

            if epoch+1==self.picture_num:
                break
        np.save(f'lime_save/lime_{self.animal}_{self.picture_num}_{config.torch_seed}.npy',image_best_save)
        np.save(f'lime_save/lime_{self.animal}_{self.picture_num}_origin_{config.torch_seed}.npy',image_origin_save)


def validate():
    image_best_save=torch.from_numpy(np.load(f'lime_save/lime_{self.animal}_{self.picture_num}_{config.torch_seed}.npy').astype(np.float32))
    image_origin_save=torch.from_numpy(np.load(f'lime_save/lime_{self.animal}_{self.picture_num}_origin_{config.torch_seed}.npy').astype(np.float32))
    print(use_model(image_best_save.cuda())[0])
    print(use_model(image_origin_save.cuda())[0])
    for i in range(20):
        img = torchvision.utils.make_grid(image_best_save[i])
        img = img.cpu().data.numpy().transpose((1, 2, 0))  # 本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
        img = img * [0.5,0.5,0.5] +[0.5,0.5,0.5]  # (228, 906, 3)范围由(-1, 1)变成(0, 1)
        plt.figure(1,figsize=(2,2),dpi=300)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(f"PPT_fig/paper/lime_{i}.pdf",dpi=300)
        plt.savefig(f"PPT_fig/paper/lime_{i}.tiff",dpi=300)
        img = torchvision.utils.make_grid(image_origin_save[i])
        img = img.cpu().data.numpy().transpose((1, 2, 0))  # 本来是(0,1,2)，相当于把第一维变为第三维，其他两维前移
        img = img * [0.5,0.5,0.5] +[0.5,0.5,0.5]  # (228, 906, 3)范围由(-1, 1)变成(0, 1)
        plt.figure(2, figsize=(2, 2), dpi=300)
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(f"PPT_fig/paper/lime_origin_{i}.pdf",dpi=300)
        plt.savefig(f"PPT_fig/paper/lime_origin_{i}.tiff",dpi=300)



Lime=lime_GA(config)
Lime.GA()
validate()



