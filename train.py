
# parser.add_argument("--img_dir1", type=str, default='D:\\PAMI2023AV\\data\\MyResizeForTrain\\frames\\')
# parser.add_argument("--img_dir2", type=str, default='D:\\PAMI2023AV\\data\\MyResizeForTest\\frames\\')

import os
import torch
import argparse
import numpy as np
import time
import shutil
import torch.optim as optim
from modelCGCN import conservativeCGCN
import torch.nn.functional as F
from utils import AverageMeter
from utils.LoadData_short import train_data_loader
import cv2
import torch.nn as nn
from loss import KLDLoss

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    '''
    parser.add_argument("--img_dir1", type=str, default='F:\\PAMI2023\\PAMI2023I\\daata\\MyResizeForTrain\\frames\\')
    parser.add_argument("--img_dir2", type=str, default='F:\\PAMI2023\\PAMI2023I\\daata\\MyResizeForTest\\frames\\')
    '''
    '''
    parser.add_argument("--img_dir1", type=str, default='D:\\PAMI2023AV\\data\\MyResizeForTrain\\frames\\')
    parser.add_argument("--img_dir2", type=str, default='D:\\PAMI2023AV\\data\\MyResizeForTest\\frames\\')
    '''
    parser.add_argument("--img_dir1", type=str, default='C:\\PAMI2023\\datasetmy\\MyResizeForTrain\\frames\\')
    parser.add_argument("--img_dir2", type=str, default='C:\\PAMI2023\\datasetmy\\MyResizeForTest\\frames\\')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=384)  ####
    parser.add_argument("--batch_size", type=int, default=6) 
    parser.add_argument("--shift_rate", type=float, default=0.7)
    parser.add_argument("--shift_thresh", type=float, default=0.4)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--snapshot_dir", type=str, default='runshh8/')
    parser.add_argument("--middir", type=str, default='runshh8/')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=5)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = conservativeCGCN(pretrained=True, shift_rate=args.shift_rate, shift_thresh=args.shift_thresh, middir=args.middir, training_epoch=args.epoch)
    device = torch.device(0)	
    model = torch.nn.DataParallel(model).cuda()
    model.to(device)
    torch.backends.cudnn.benchmark=True
    param_groups = model.module.get_parameter_groups()
    '''optimizer = torch.optim.AdamW([
        {'params': param_groups[0], 'lr': 1*args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], args.lr, weight_decay=1e-4)'''
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': 1*args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    return model, optimizer

def train(args, save_index):
    criteria = nn.BCEWithLogitsLoss().cuda()
    criterion = KLDLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter

    train_loader, val_loader = train_data_loader(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    
    model.train()
    print(model)
    end = time.time()
    current_epoch = 0

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        steps_per_epoch = len(train_loader)

        index = 0  
        for idx, dat in enumerate(train_loader):
            
            image1,image,image2,salmap1,salmap,salmap2,salmap10,salmap0,salmap20,path,weight,weight1 = dat

            image = image.cuda(non_blocking=True)
            image1 = image1.cuda(non_blocking=True)
            image2 = image2.cuda(non_blocking=True)  

            salmap = salmap.cuda(non_blocking=True)
            salmap1 = salmap1.cuda(non_blocking=True)
            salmap2 = salmap2.cuda(non_blocking=True) 

            salmap0 = salmap0.cuda(non_blocking=True)
            salmap10 = salmap10.cuda(non_blocking=True)
            salmap20 = salmap20.cuda(non_blocking=True) 

            weight = weight.cuda(non_blocking=True)
            weight1 = weight1.cuda(non_blocking=True)  
            
            x1, x2, x3, x4, x5, x6, weightc, weightc1 = model(idx, image1, image, image2)
            # loss_train = criterion(x1, salmap)+criterion(x2, salmap1)+criterion(x3, salmap2)
            # loss_train = loss_train / 4

            nanarray = torch.argwhere(torch.isnan(weight))
            for nanindex in range(nanarray.shape[0]):
                weight[nanarray[nanindex]] = torch.zeros(1).cuda()
            nanarray = torch.argwhere(torch.isnan(weight1))
            for nanindex in range(nanarray.shape[0]):
                weight1[nanarray[nanindex]] = torch.zeros(1).cuda()

            loss_train = torch.zeros(1).cuda()
            for mm in range(0, image.size(0)):
                loss_train1 = criterion(x1[mm], (1+weight[mm]*salmap10[mm]) *salmap1[mm]) 
                if torch.isnan(loss_train1):
                    loss_train1 = torch.zeros(1).cuda()
                loss_train2 = criterion(x2[mm], (1+max(weight[mm],weight1[mm])*salmap0[mm]) *salmap[mm])
                if torch.isnan(loss_train2):
                    loss_train2 = torch.zeros(1).cuda()
                loss_train3 = criterion(x3[mm], (1+weight1[mm]*salmap20[mm]) *salmap2[mm])
                if torch.isnan(loss_train3):
                    loss_train3 = torch.zeros(1).cuda()

                loss_train4 = criterion(x4[mm], (1+weight[mm]*salmap10[mm]) *salmap1[mm]) 
                if torch.isnan(loss_train4):
                    loss_train4 = torch.zeros(1).cuda()
                loss_train5 = criterion(x5[mm], (1+max(weight[mm]+weight1[mm])*salmap0[mm]) *salmap[mm]) 
                if torch.isnan(loss_train5):
                    loss_train5 = torch.zeros(1).cuda()
                loss_train6 = criterion(x6[mm], (1+weight1[mm]*salmap20[mm]) *salmap2[mm]) 
                if torch.isnan(loss_train6):
                    loss_train6 = torch.zeros(1).cuda()
                loss_train = loss_train + (loss_train1+loss_train2+loss_train3+loss_train4+loss_train5+loss_train6) +5*(F.mse_loss(weight,weightc)+F.mse_loss(weight1,weightc1))
            
            loss_train = loss_train / image.size(0)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), image.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))
            if global_counter % 200 == 0:
                save_index = save_index + 1
                with torch.no_grad():

                    correct = 0
                    total = 0

                    for idx_test, dat_test in enumerate(val_loader):
                        model.eval()
                        image1,image,image2,path = dat_test
                        image = image.cuda(non_blocking=True)
                        image1 = image1.cuda(non_blocking=True)
                        image2 = image2.cuda(non_blocking=True)   

                        x1, x2, x3, x4, x5,x6,w1, w2 = model(idx_test, image1, image, image2)
                        batch_num = image.size()[0]  # 3

                        for ii in range(batch_num):
                            imgp = path[ii].split('\\')[-2]# path[ii].split('/')[-2]
                            imgn = path[ii].split('\\')[-1]# path[ii].split('/')[-1]
                            featpath = './runshh8/'

                            accu_map_name = os.path.join(featpath, str(save_index), imgp, imgn)
                            if not os.path.exists(os.path.join(featpath, str(save_index))):
                                os.mkdir(os.path.join(featpath, str(save_index)))
                            if not os.path.exists(os.path.join(featpath, str(save_index))):
                                os.mkdir(os.path.join(featpath, str(save_index)))
                            if not os.path.exists(os.path.join(featpath, str(save_index), imgp)):
                                os.mkdir(os.path.join(featpath, str(save_index), imgp))

                            atts = x2[ii][0]
                            # atts[atts < 0] = 0
                            att = atts.cpu().data.numpy()
                            att = np.rint(att / (att.max() + 1e-8) * 255)
                            att = np.array(att, np.uint8)
                            att = cv2.resize(att, (360, 180))
                            cv2.imwrite(accu_map_name[:-4]+'_c.jpg', att)

                            heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                            img = cv2.imread(path[ii])
                            img = cv2.resize(img, (360, 180))
                            result = heatmap * 0.3 + img * 0.5
                            cv2.imwrite(accu_map_name, result)
                print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

                savepath = os.path.join('./model/', str(save_index)+'.pth')
                torch.save(model.state_dict(), savepath)
                model.train()
        if current_epoch == args.epoch-1:
            save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        }, is_best=False,
                        filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    save_index = 0
    train(args, save_index)
