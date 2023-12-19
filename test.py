
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
from utils.LoadData_short import test_data_loader2
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
    parser.add_argument("--img_dir1", type=str, default='C:\\PAMI2023\\datasetmy\\MyResizeForTrain\\frames\\')
    parser.add_argument("--img_dir2", type=str, default='C:\\PAMI2023\\datasetmy\\MyResizeForTest\\frames\\')
    #'''
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=384)  ####
    parser.add_argument("--batch_size", type=int, default=3) 
    parser.add_argument("--shift_rate", type=float, default=0.7)
    parser.add_argument("--shift_thresh", type=float, default=0.4)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--snapshot_dir", type=str, default='runs/')
    parser.add_argument("--middir", type=str, default='runs/')
    parser.add_argument("--num_workers", type=int, default=0)
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
    checkpoint = torch.load('./model/18.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
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
import time
def test(args):
    model, optimizer = get_model(args)
    val_loader = test_data_loader2(args)
    save_idx = 0
    with torch.no_grad():
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
                featpath = './result12/'

                
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


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    save_index = 0
    test(args)


