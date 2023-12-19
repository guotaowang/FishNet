from .transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import random

def train_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225] 
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.RandomCrop(crop_size),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDatasetT(root_dir=args.img_dir1, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDatasetE(root_dir=args.img_dir2, num_classes=args.num_classes, transform=tsfm_test, test=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=3, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def test_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDatasetE(root_dir=args.img_dir2, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDatasetT(Dataset):
    def __init__(self, root_dir, num_classes=28, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.name_list, self.figs_list, self.seqs_list = self.read_labeled_image_list_train(self.root_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):    
        pathimla = self.image_list[idx]
        img_la = pathimla.split('+')

        img_name = img_la[0]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        inda = int(img_la[1])
        pathname = self.name_list[inda]
        images = []
        image_names = []

        for hh in range(-1,2):
            aa = self.seqs_list[inda].index(img_la[4])  # 同视频段
            bb = random.randrange(1, pathname[aa][1]) # 随机帧
            pick_path = os.path.join(img_la[2], img_la[3], pathname[aa][0], self.figs_list[inda][aa][bb][:-4]+'.jpg')
            image_pick = Image.open(pick_path).convert('RGB')
            image_pick = self.transform(image_pick)
            image_names.append(pick_path)
            images.append(image_pick)
            
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return img_name, image, inda, label, image_names[0], images[0], inda, label, image_names[1], images[1], inda, label


    def read_labeled_image_list_train(self, data_dir):
        path_list = []
        name_list = []
        figs_list = []
        seqs_list = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, 1):# len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            num_list = []
            fig_list = []
            seq_list = []
            for fs in range(0, len(ficname)):
                picpath = os.path.join(ficpath, ficname[fs])
                picname = os.listdir(picpath)
                picname.sort()
                if len(picname) < 3:
                    continue
                for picp in range(1, len(picname)-1):
                    if picname[picp].endswith('.jpg'):
                        pv1 = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp])
                        path_list.append(pv1+'+'+str(file)+'+'+data_dir+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
                num_list.append([ficname[fs], len(picname)])
                fig_list.append(picname)
                seq_list.append(ficname[fs])
            name_list.append(num_list)
            figs_list.append(fig_list)
            seqs_list.append(seq_list)
        return path_list, name_list, figs_list, seqs_list

class VOCDatasetE(Dataset):
    def __init__(self, root_dir, num_classes=28, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.name_list, self.figs_list, self.seqs_list = self.read_labeled_image_list_test(self.root_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):    
        pathimla = self.image_list[idx]
        img_la = pathimla.split('+')

        img_name = img_la[0]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        inda = int(img_la[1])
        pathname = self.name_list[inda]
        images = []
        image_names = []

        for hh in range(-1,2):
            aa = self.seqs_list[inda].index(img_la[4])  # 同视频段
            bb = random.randrange(1, pathname[aa][1]) # 随机找帧
            pick_path = os.path.join(img_la[2], img_la[3], pathname[aa][0], self.figs_list[inda][aa][bb][:-4]+'.jpg')
            image_pick = Image.open(pick_path).convert('RGB')
            image_pick = self.transform(image_pick)
            image_names.append(pick_path)
            images.append(image_pick)
            
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return img_name, image, inda, label, image_names[0], images[0], inda, label, image_names[1], images[1], inda, label

    def read_labeled_image_list_test(self, data_dir):
        path_list = []
        name_list = []
        figs_list = []
        seqs_list = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            num_list = []
            fig_list = []
            seq_list = []
            for fs in range(0, len(ficname)):
                picpath = os.path.join(ficpath, ficname[fs])
                picname = os.listdir(picpath)
                picname.sort()
                if len(picname) < 3:
                    continue
                picnamen = []
                for picp in range(2, len(picname)-2, 30):
                    if picname[picp].endswith('.jpg'):
                        pv1 = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp])
                        path_list.append(pv1+'+'+str(file)+'+'+data_dir+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
                        picnamen.append(picname[picp])
                num_list.append([ficname[fs], len(picnamen)])
                fig_list.append(picnamen)
                seq_list.append(ficname[fs])
            name_list.append(num_list)
            figs_list.append(fig_list)
            seqs_list.append(seq_list)
        return path_list, name_list, figs_list, seqs_list

    
