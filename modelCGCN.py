import torch.nn as nn
import torch
import torch.nn.functional as F
from ConvGRU import ConvGRUCell
import os
from lib.pvtv2 import pvt_small
from trans_utils.transformer import build_transformer
from trans_utils.position_encoding import build_position_encoding
import torch.utils.model_zoo as model_zoo
from util import remove_layer
from lib.SphereConv2d import SphereConv2d 
# from torchvision.ops import DeformConv2d
from lib.geometry import *
from Soundmodel import SoundNet
from dcn import DeformableConv2d
from scipy import ndimage

model_urls = {'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

off = []
off.append(torch.from_numpy(compute_deform_offset(4, 384)).float().cuda()) # 4*4*2=32
off.append(torch.from_numpy(compute_deform_offset(2, 384//4)).float().cuda()) # 2*2*2=8
off.append(torch.from_numpy(compute_deform_offset(2, 384//8)).float().cuda())
off.append(torch.from_numpy(compute_deform_offset(2, 384//16)).float().cuda())

dst = []# 32, 384
dst.append(visualize_patch2(8, 384//4).cuda()) # 96
dst.append(visualize_patch2(4, 384//8).cuda()) # 48
dst.append(visualize_patch2(2, 384//16).cuda()) # 24
# dst.append(visualize_patch2(1, 384//32).cuda()) # 12

def normalize_tensor(In, range_min=0, range_max=1):
    batch_size = In.size(0)
    normalized_InALL = []

    for i in range(batch_size):
        min_values = torch.min(In[i, 0])
        max_values = torch.max(In[i, 0])

        normalized_In = (In[i, 0] - min_values) / (max_values - min_values)
        normalized_In = normalized_In * (range_max - range_min) + range_min

        normalized_InALL.append(normalized_In)
    normalized_InALL = torch.stack(normalized_InALL, dim=0).unsqueeze(1)
    return normalized_InALL

def spherical_distance(theta1, phi1, theta2, phi2):
    cos_distance = torch.sin(theta1)*torch.sin(theta2)+torch.cos(theta1)*torch.cos(theta2)*torch.cos(phi1-phi2)
    distance = torch.arccos(cos_distance) / torch.tensor([np.pi], dtype=torch.float32).cuda()
    return distance

class AttShiftW(nn.Module):
    def __init__(self):
        super(AttShiftW, self).__init__()

    def forward(self, input_1,input_2,input_3):
        self.extra_lrelu = nn.LeakyReLU()
        theta_1, phi_1, rMask_1, cMean_1 = self._findMax(input_1)
        theta_2, phi_2, rMask_2, cMean_2 = self._findMax(input_2)
        theta_3, phi_3, rMask_3, cMean_3 = self._findMax(input_3)
        weight1 = spherical_distance(theta_1, phi_1, theta_2, phi_2).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        nanarray = torch.argwhere(torch.isnan(weight1))
        for nanindex in range(nanarray.shape[0]):
            weight1[nanarray[nanindex]] = torch.zeros(1).cuda()

        weight2 = spherical_distance(theta_2, phi_2, theta_3, phi_3).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        nanarray = torch.argwhere(torch.isnan(weight2))
        for nanindex in range(nanarray.shape[0]):
            weight2[nanarray[nanindex]] = torch.zeros(1).cuda()

        return weight1,weight2,rMask_1,rMask_2,rMask_3

    def _findMax(self, in_):
        shift_thresh = 0.4
        cMean = normalize_tensor(torch.mean(in_, dim=1, keepdim=True)) # 1,1,32,32  ; 48,96
        width = cMean.size(3)
        height = cMean.size(2)
        
        batch_size = cMean.size(0)
        maxval, _ = torch.max(cMean.view(batch_size, -1), dim=1, keepdim=True)
        throld = maxval * shift_thresh #torch.normal(mean=torch.tensor([0.6]), std=torch.tensor([0.01]))[0]
        throld = throld.view(batch_size, 1, 1, 1)

        rMask = (cMean >= throld).float()

        labeled_matrix, num_features = ndimage.label(rMask.cpu()) # 3,1,48,96

        theta = []
        phi = []
        result = torch.zeros_like(cMean).cuda()
        indices = torch.nonzero(cMean >= throld)

        for i in range(batch_size):
            region_sizes = torch.bincount(torch.tensor(labeled_matrix[i].reshape(-1)))[1:]
            if region_sizes.numel()>0:
                largest_region_label = torch.argmax(region_sizes) + 1
                largest_region_indices = (torch.tensor(labeled_matrix[i]) == largest_region_label).nonzero(as_tuple=False)
                result[i,0,largest_region_indices.long()[:,1],largest_region_indices.long()[:,2]] = cMean[i,0,largest_region_indices.long()[:,1],largest_region_indices.long()[:,2]]
                phi.append(torch.mean(largest_region_indices.float()[:,1]).cuda())
                theta.append(torch.mean(largest_region_indices.float()[:,2]).cuda())
            else:
                mask = (indices[:, 0] == i) # 7424,4
                phi.append(torch.mean(indices[mask, :].float()[:,2]))
                theta.append(torch.mean(indices[mask, :].float()[:,3]))
        phi = (0.5- torch.stack(phi) / height) * np.pi
        theta = (torch.stack(theta) / width - 0.5) * 2 * np.pi

        return theta.cuda(), phi.cuda(), result, cMean


class CGCN(nn.Module):
    def  __init__(self, features, all_channel=32, att_dir='./runs/', training_epoch=10,**kwargs):
        super(CGCN, self).__init__()
        all_channel = 32
        self.extra_convs = nn.Sequential(nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel*3, kernel_size=3, padding=1), nn.BatchNorm2d(all_channel*3), nn.ReLU(inplace=True), 
                                        nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel*3, kernel_size=3, padding=1), nn.BatchNorm2d(all_channel*3), nn.ReLU(inplace=True), 
                                        nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel*3, kernel_size=3, padding=1), nn.BatchNorm2d(all_channel*3), nn.ReLU(inplace=True), 
                                        nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel, kernel_size=1, padding=0))

        self.channel = all_channel
        self.extra_AttShiftW = AttShiftW()
        
        d = self.channel // 2

        self.extra_projf = nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel*3 // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel*3 // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel*3, out_channels=all_channel*3, kernel_size=1)
        self.extra_gates = nn.Conv2d(all_channel*3, 1, kernel_size = 1, bias = False)
        self.extra_gates_s = nn.Sigmoid()

        self.extra_conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=1, padding=0, bias=False)
        self.extra_softmax = nn.Sigmoid()
        self.extra_prelu = nn.ReLU(inplace=True)
        self.extra_bn1 = nn.BatchNorm2d(all_channel)
        self.extra_main_classifier1 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=True)
        self.extra_main_classifier2 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=True)

        self.extra_Trl2 = nn.Sequential(DeformableConv2d(128, all_channel, kernel_size=1, padding=0), nn.BatchNorm2d(all_channel))
        self.extra_Trl3 = nn.Sequential(DeformableConv2d(320, all_channel, kernel_size=1, padding=0), nn.BatchNorm2d(all_channel))
        self.extra_Trl4 = nn.Sequential(DeformableConv2d(512, all_channel, kernel_size=1, padding=0), nn.BatchNorm2d(all_channel))

        self.positional_embedding = build_position_encoding(hidden_dim=all_channel, mode='learned')
        self.transformer = build_transformer(d_model=all_channel*3, nhead=1, num_encoder_layers=3)
        num_queries = 1
        self.query_embed = nn.Embedding(num_queries, all_channel)

        self.extra_refineST = nn.Sequential(nn.Conv3d(all_channel, all_channel, (3, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(all_channel), nn.ReLU(inplace=True), 
                                            nn.Conv3d(all_channel, all_channel, (1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(all_channel), nn.ReLU(inplace=True),
                                            nn.Conv3d(all_channel, all_channel, (1, 3, 3), padding=(0, 1, 1)),
                                            )

        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  

        self.features = features
    		
    def forward(self, idx,  input1, input2, input3):
        B, C, H, W = input1.shape
 
        x1 = self.features(input1) # 1,512,12,24,## 512,320,128,64
        x1_2 = self.extra_Trl2(F.grid_sample(x1[1], dst[1].repeat(B, 1, 1, 1), align_corners=False))
        x1_3 = self.extra_Trl3(F.grid_sample(x1[2], dst[2].repeat(B, 1, 1, 1), align_corners=False))
        x1_4 = self.extra_Trl4(x1[3])
        x1_4 = F.upsample(x1_4, size=x1_2.size()[2:], mode='bilinear')
        x1_3 = F.upsample(x1_3, size=x1_2.size()[2:], mode='bilinear')
        x1_c = self.extra_convs(torch.cat((x1_4,x1_3,x1_2),1))

        x2 = self.features(input2) # 1,512,32,32
        # x2_1 = self.extra_Trl1(x2[0])
        x2_2 = self.extra_Trl2(F.grid_sample(x2[1], dst[1].repeat(B, 1, 1, 1), align_corners=False))
        x2_3 = self.extra_Trl3(F.grid_sample(x2[2], dst[2].repeat(B, 1, 1, 1), align_corners=False))
        x2_4 = self.extra_Trl4(x2[3])
        x2_4 = F.upsample(x2_4, size=x2_2.size()[2:], mode='bilinear')
        x2_3 = F.upsample(x2_3, size=x2_2.size()[2:], mode='bilinear')
        x2_c = self.extra_convs(torch.cat((x2_4,x2_3,x2_2),1))    

        x3 = self.features(input3) # 1,512,32,32
        # x3_1 = self.extra_Trl1(x3[0])
        x3_2 = self.extra_Trl2(F.grid_sample(x3[1], dst[1].repeat(B, 1, 1, 1), align_corners=False))
        x3_3 = self.extra_Trl3(F.grid_sample(x3[2], dst[2].repeat(B, 1, 1, 1), align_corners=False))
        x3_4 = self.extra_Trl4(x3[3])
        x3_4 = F.upsample(x3_4, size=x3_2.size()[2:], mode='bilinear')
        x3_3 = F.upsample(x3_3, size=x3_2.size()[2:], mode='bilinear')
        x3_c = self.extra_convs(torch.cat((x3_4,x3_3,x3_2),1))

        weight1,weight2,rMask_1,rMask_2,rMask_3 = self.extra_AttShiftW(x1_c, x2_c, x3_c)

        incat1 = torch.cat((x3_c.unsqueeze(1), x1_c.unsqueeze(1), x2_c.unsqueeze(1)), 1).view(B, 32, 3, x1_c.size(2), x1_c.size(3))
        incat2 = torch.cat((x1_c.unsqueeze(1), x2_c.unsqueeze(1), x3_c.unsqueeze(1)), 1).view(B, 32, 3, x2_c.size(2), x2_c.size(3))
        incat3 = torch.cat((x2_c.unsqueeze(1), x3_c.unsqueeze(1), x1_c.unsqueeze(1)), 1).view(B, 32, 3, x3_c.size(2), x3_c.size(3))

        st1 = self.extra_refineST(incat1).squeeze(2)
        st2 = self.extra_refineST(incat2).squeeze(2)
        st3 = self.extra_refineST(incat3).squeeze(2)

        # STAtrans = torch.cat((F.relu((1+(1+weight1)*rMask_1)*st1), F.relu((1+(1+max(weight1+weight2))*rMask_2)*st2), F.relu((1+(1+weight2)*rMask_3)*st3)),1) # 4,96,48,96
        STAtrans = torch.cat(((1+weight1*rMask_1)*st1, (1+max(weight1+weight2)*rMask_2)*st2, (1+weight2*rMask_3)*st3),1) # 4,96,48,96

        feat = self.self_attention(STAtrans)
        h_v1 = feat[:,0:32,:,:]
        h_v2 = feat[:,32:32*2,:,:]
        h_v3 = feat[:,32*2:32*3,:,:]

        attention1 = self.my_fcn(h_v1, x1_c, [192,384])#input1.size()[2:])
        attention2 = self.my_fcn(h_v2, x2_c, [192,384])#input1.size()[2:])
        attention3 = self.my_fcn(h_v3, x3_c, [192,384])#input1.size()[2:])

        attention4 = self.my_fcn2(x1_c, [192,384])#input1.size()[2:])
        attention5 = self.my_fcn2(x2_c, [192,384])#input1.size()[2:])
        attention6 = self.my_fcn2(x3_c, [192,384])#input1.size()[2:])

        return attention1,attention2,attention3,attention4,attention5,attention6,weight1,weight2

    def self_attention(self, x):
        m_batchsize, C, width, height = x.size()  # 8,32,8,8
        f = self.extra_projf(x).view(m_batchsize, -1, width * height)
        g = self.extra_projg(x).view(m_batchsize, -1, width * height)
        h = self.extra_projh(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = F.softmax(attention, dim=1)  # 8,1024,1024

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H
        self_mask = self.extra_gates(self_attetion)  # [1, 1, 8, 8]
        self_mask = self.extra_gates_s(self_mask)
        out = self_mask * x
        return out

 
    def my_fcn(self, input1_att,  exemplar,  input_size):  # exemplar,

        input1_att = torch.cat([input1_att, exemplar], 1)  # [1, 512, 45, 45]
        input1_att = self.extra_conv1(input1_att)            # [1, 256, 45, 45]
        input1_att = self.extra_bn1(input1_att)
        input1_att = self.extra_prelu(input1_att)
        x1 = self.extra_main_classifier1(input1_att)           # [1, 256, 45, 45]
        x1 = F.sigmoid(x1)
        x1 = F.upsample(x1, input_size, mode='bilinear')  # [1, 1, 354, 354]
        return x1  # , x2, temp  #shape: NxCx

    def my_fcn2(self, input1_att,  input_size):  # exemplar,
        x1 = self.extra_main_classifier2(input1_att)           # [1, 256, 45, 45]
        x1 = F.sigmoid(x1)
        x1 = F.upsample(x1, input_size, mode='bilinear')  # [1, 1, 354, 354]
        return x1  # , x2, temp  #shape: NxCx


    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


def conservativeCGCN(pretrained=True, **kwargs):
    backbone = pvt_small()  # [64, 128, 320, 512]
    path = './pretrained_pth/pvt_small.pth'
    save_model = torch.load(path)
    model_dict = backbone.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    backbone.load_state_dict(model_dict)

    model = CGCN(backbone, **kwargs)

    return model
