from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import cv2
import torch
from lib.geometry import *
import torch.nn.functional as F
dst = visualize_patch(32, 384)
dst2 = visualize_patch2(32, 384)
dst3 = visualize_patch3(32, 384)

import math

def _findMax(in_):
    indices = torch.nonzero(in_)
    theta = []
    phi = []
    for i in range(in_.size(0)):
        mask = (indices[:, 0] == i)
        phi.append(torch.mean(indices[mask, :].float()[:,1]))
        theta.append(torch.mean(indices[mask, :].float()[:,2]))
    phi = (0.5- torch.stack(phi) / in_.size(1)) * np.pi
    theta = (torch.stack(theta) / in_.size(2) - 0.5) * 2 * np.pi

    return theta, phi

def spherical_distance(theta1, phi1, theta2, phi2):
    cos_distance = torch.sin(theta1)*torch.sin(theta2)+torch.cos(theta1)*torch.cos(theta2)*torch.cos(phi1-phi2)
    distance = torch.arccos(cos_distance) / torch.tensor([np.pi], dtype=torch.float32)
    return distance

def train_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225] 
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize((input_size, input_size*2)),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    tsfm_grount = transforms.Compose([transforms.Resize((192, 192*2)), 
                                    transforms.ToTensor()
                                    ])

    tsfm_test = transforms.Compose([transforms.Resize((input_size, input_size*2)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDatasetT(root_dir=args.img_dir1, transform_im=tsfm_train,transform_gt=tsfm_grount, test=False)
    img_test = VOCDatasetE(root_dir=args.img_dir2, transform_im=tsfm_test, test=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)
    val_loader = DataLoader(img_test, batch_size=3, shuffle=False, num_workers=0)

    return train_loader, val_loader


def test_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize((input_size, input_size*2)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDatasetE(root_dir=args.img_dir2, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDatasetT(Dataset):
    def __init__(self, root_dir, num_classes=309, transform_im=None, transform_gt=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.transform_im = transform_im
        self.transform_gt = transform_gt
        self.num_classes = num_classes
        self.inputs = self.read_labeled_image_list_train(self.root_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):    
        # Load i-th image
        img_path = self.inputs[idx]
        image = Image.open(img_path).convert('RGB')

        image = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))
          
        num1 = random.randrange(-15,0,1)
        img_path1 = self.inputs[idx][:-8]+("%04d" % (int(self.inputs[idx][-8:-4])+num1))+'.jpg'
        image1 = Image.open(img_path1).convert('RGB')
        image1 = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image1),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))

        num2 = random.randrange(0,15,1)
        img_path2 = self.inputs[idx][:-8]+("%04d" %  (int(self.inputs[idx][-8:-4])+num2))+'.jpg'
        image2 = Image.open(img_path2).convert('RGB')
        image2 = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image2),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))

        sal_path = img_path.replace('frames', 'fixResultResizeGauss')[:-4]+'.png'
        salmap = Image.open(sal_path).convert('L')
        sal_path0 = img_path.replace('frames', 'fixResultResizeGaussShift')[:-4]+'.png'
        salmap0 = Image.open(sal_path0).convert('L')

        sal_path1 = img_path.replace('frames', 'fixResultResizeGauss')[:-4]+'.png'
        salmap1 = Image.open(sal_path1).convert('L')
        sal_path10 = img_path1.replace('frames', 'fixResultResizeGaussShift')[:-4]+'.png'
        salmap10 = Image.open(sal_path10).convert('L')

        sal_path2 = img_path1.replace('frames', 'fixResultResizeGauss')[:-4]+'.png'
        salmap2 = Image.open(sal_path2).convert('L')
        sal_path20 = img_path2.replace('frames', 'fixResultResizeGaussShift')[:-4]+'.png'
        salmap20 = Image.open(sal_path20).convert('L')

        image = self.transform_im(image)
        image1 = self.transform_im(image1)
        image2 = self.transform_im(image2)

        salmap = self.transform_gt(salmap)
        salmap1 = self.transform_gt(salmap1)
        salmap2 = self.transform_gt(salmap2)

        salmap0 = self.transform_gt(salmap0)
        salmap10 = self.transform_gt(salmap10)
        salmap20 = self.transform_gt(salmap20)
        weight = spherical_distance(_findMax(salmap10)[0],_findMax(salmap10)[1],_findMax(salmap0)[0],_findMax(salmap0)[1])
        weight1 = spherical_distance(_findMax(salmap0)[0],_findMax(salmap0)[1],_findMax(salmap20)[0],_findMax(salmap20)[1])


        path = img_path
        return image1,image,image2,salmap1,salmap,salmap2,salmap10,salmap0,salmap20,path,weight,weight1

    def read_labeled_image_list_train(self, data_dir):
        path_list = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            for fs in range(15, len(ficname)-15):
                picpath = os.path.join(ficpath, ficname[fs])
                if ficname[fs].endswith('.jpg'):
                    pv1 = os.path.join(picpath)
                    path_list.append(pv1)
        return path_list

class VOCDatasetE(Dataset):
    def __init__(self, root_dir, num_classes=309, transform_im=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.transform_im = transform_im
        self.num_classes = num_classes
        self.inputs = self.read_labeled_image_list_test(self.root_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):    
        img_path = self.inputs[idx]
        image = Image.open(img_path).convert('RGB')
        image = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))
          
        num1 = 1
        img_path1 = self.inputs[idx][:-8]+("%04d" % (int(self.inputs[idx][-8:-4])+num1))+'.jpg'
        image1 = Image.open(img_path1).convert('RGB')
        image1 = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image1),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))

        num2 = -1
        img_path2 = self.inputs[idx][:-8]+("%04d" %  (int(self.inputs[idx][-8:-4])+num2))+'.jpg'
        image2 = Image.open(img_path2).convert('RGB')
        image2 = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image2),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))

        image = self.transform_im(image)
        image1 = self.transform_im(image1)
        image2 = self.transform_im(image2)

        path = img_path
        return image1,image,image2,path

    def read_labeled_image_list_test(self, data_dir):
        path_list = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            for fs in range(15, len(ficname)-15,10):
                picpath = os.path.join(ficpath, ficname[fs])
                if ficname[fs].endswith('.jpg'):
                    pv1 = os.path.join(picpath)
                    path_list.append(pv1)
        return path_list
    
def test_data_loader2(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize((input_size, input_size*2)),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDatasetE2(root_dir=args.img_dir2, transform_im=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDatasetE2(Dataset):
    def __init__(self, root_dir, num_classes=309, transform_im=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.transform_im = transform_im
        self.num_classes = num_classes
        self.inputs = self.read_labeled_image_list_test2(self.root_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):    
        img_path = self.inputs[idx]
        image = Image.open(img_path).convert('RGB')
        image = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))
          
        num1 = 1
        img_path1 = self.inputs[idx][:-8]+("%04d" % (int(self.inputs[idx][-8:-4])+num1))+'.jpg'
        image1 = Image.open(img_path1).convert('RGB')
        image1 = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image1),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))

        num2 = -1
        img_path2 = self.inputs[idx][:-8]+("%04d" %  (int(self.inputs[idx][-8:-4])+num2))+'.jpg'
        image2 = Image.open(img_path2).convert('RGB')
        image2 = Image.fromarray(cv2.remap(cv2.resize(np.asarray(image2),(768,384)), dst[:,:,1].astype(np.float32), dst[:,:,0].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP))

        image = self.transform_im(image)
        image1 = self.transform_im(image1)
        image2 = self.transform_im(image2)

        path = img_path
        return image1,image,image2,path

    def read_labeled_image_list_test2(self, data_dir):
        path_list = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            for fs in range(1, len(ficname)-1): #  len(ficname)-1
                picpath = os.path.join(ficpath, ficname[fs])
                if ficname[fs].endswith('.jpg'):
                    pv1 = os.path.join(picpath)
                    path_list.append(pv1)
        return path_list
    
