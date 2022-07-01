import torch
import os, re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch.optim as optim
import  torch.nn as nn
import numpy as np
import scipy.io as sio
import argparse
from tqdm import tqdm
import cvxpy as cp
import time
import matplotlib.pyplot as plot

from model.TV_denoising import TV_denoising, TV_denoising3d
from utils.utils import clip, ssim, psnr
#from utils.ani import save_ani
from model.network_ffdnet import FFDNet
from model.network_fastdvd import FastDVDnet

from Cvxpy_Solver import Alternate_solution

parser = argparse.ArgumentParser(description='Select device')
parser.add_argument('--device', default=0)
# parser.add_argument('--level', default=0)
args = parser.parse_args()
device_num = args.device
# level = float(args.level)
device = 'cuda:{}'.format(device_num)
torch.no_grad()
model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(device)
model.load_state_dict(torch.load('pretrained_models/ffdnet_gray.pth'))
model.eval()

model_FastDVD = FastDVDnet(num_input_frames= 5,num_color_channels=1)
model_FastDVD = model_FastDVD.cuda()
state_temp_dict = torch.load('pretrained_models/model_gray.pth')
model_FastDVD.load_state_dict(state_temp_dict)

from PyDRUNet.drunet_denoise import DRUNet_Denoise

cost = torch.nn.MSELoss()

def A(x,Phi):       #N*C*W*H
    temp = x*Phi
    y = torch.sum(temp,2,keepdim=False)
    return y

def At(dy,Phi): #N*C*W*H
    #temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    temp = dy.unsqueeze(2).repeat(1,1,Phi.shape[2])
    dx = temp*Phi
    return dx

def ffdnet_denosing(x, sigma, netvnorm_init):
    image_m, image_n, image_c = x.shape
    for k, v in model.named_parameters():
        v.requires_grad = False
    if netvnorm_init:
        x_min = x.min().item()
        x_max = x.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        x = (x - x_min) / (x_max - x_min)
        x = x * scale + shift
        sigma = torch.tensor(sigma / (x_max - x_min) * scale, device=device)
    else:
        sigma = torch.tensor(sigma, device=device)

    frame_list = []
    with torch.no_grad():
        for j in range(image_c):
            temp_x = x[:, :, j].view(1, 1, image_m, image_n)
            estimate_img = model(temp_x, sigma.view(1, 1, 1, 1))
            frame_list.append(estimate_img[0, 0, :, :])
        x = torch.stack(frame_list, dim=2)

    if netvnorm_init:
        x = (x - shift) / scale
        x = x * (x_max - x_min) + x_min
    return x

def fastdvdnet_denosing(x, sigma, netvnorm_init):
    image_m, image_n, image_c = x.shape
    if netvnorm_init:
        x_min = x.min().item()
        x_max = x.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        x = (x - x_min) / (x_max - x_min)
        x = x * scale + shift
        sigma = torch.tensor(sigma / (x_max - x_min) * scale, device=device)
    else:
        sigma = torch.tensor(sigma, device=device)

    x = x.unsqueeze(3) # unsqueeze the color dimension - [H,W,F] to [H,W,F,C=1]
    x = x.permute(2, 3, 0, 1) # from H x W x F x C to F x C x H x W
    
    windsize = 5
    with torch.no_grad():
        hw = int((windsize-1)//2) # half window size
        
        N, C, H, W = x.shape
        x_denoised = torch.empty((N, C, H, W)).cuda()
        sigma_map = sigma.expand(1, 1, H, W).cuda()
        
        for frameidx in range(N):
            idx = torch.tensor(range(frameidx, frameidx+windsize)) - hw
            # % N # circular padding Changed!!!!
            
            idx = idx.clamp(0, N-1)
            x_seq = x[idx].reshape((1, -1, H, W)) # reshape from [W, C, H, W] to [1, W*C, H, W]

            x_denoised[frameidx] = model_FastDVD(x_seq, sigma_map)

    x = x_denoised.permute(2, 3, 0, 1).squeeze(3) # back from F x C=1 x H x W to H x W x F
    
    if netvnorm_init:
        x = (x - shift) / scale
        x = x * (x_max - x_min) + x_min
    return x

def run(mat_file):
    mat_data = sio.loadmat(mat_file)
    im_orig = mat_data['orig'] / 1
    im_orig = torch.from_numpy(im_orig).type(torch.float32).to(device)
    image_m, image_n, image_c = im_orig.shape
    # image_seq = []
    # ---- load mask matrix ----
    mask_A = torch.from_numpy(mat_data['mask'].astype(np.float32)).to(device)
    y = mat_data['meas'] / 1
    y = torch.from_numpy(y).type(torch.float32).to(device)
 
    mask_sum_A = torch.sum(mask_A**2, dim=2, keepdim=False)
    mask_sum_A[mask_sum_A == 0] = 1
    
    try:
        nChannel = y.size(2) 
    except: 
        nChannel=1
    
    netvnorm_init = False#True
    acc = False
  
    if NoMuilt:     # Channel
        nChannel=1  # Single 
    x = At(torch.div(y[:,:,1],mask_sum_A[:,:,1]), mask_A[:,:,:,1])   # dimension - [H,W,F]
    v = x
    u = torch.zeros_like(x, dtype=torch.float32, device=device)
    
    u_list = []
    for iC in range(nChannel):
        u_list.append(u)
        
    y1 = torch.zeros_like(y, dtype=torch.float32, device=device)
    
    v_old = v
    delta_v_old = 1
    sigma = 50.
    i_old = 0
    
    try:
        min_sigma = float(re.findall(r'\d+', mat_file)[-1])   # Find the level of Noise
    except:
        min_sigma = 5
    
    if min_sigma < 5:
        min_sigma = 5

    for i in tqdm(range(100)):
        if i == 20: netvnorm_init= False
        
        ffdnet_mat_list = []
        ffdnet_list_list = []
        
        x_list = []
        v_list = []
        
        
        for iC in range(nChannel):
            yy = y[:,:,iC]
            mask = mask_A[:,:,:,iC]
            mask_sum = mask_sum_A[:,:,iC]
            u = u_list[iC]
            
            yb = A(v + u, mask)

            if acc:
                y1 = y1 + (yy - yb)
                temp = (y1 - yb) / mask_sum
            else:
                y1 = yy - yb
                temp = y1 / mask_sum
            x = v + 1 * At(temp, mask)
            x_list.append(x)

            if i < 10:
                #v = ffdnet_denosing(x+u, 50./255, netvnorm_init)
                v = TV_denoising3d(x+u, 1., 7).clamp(0, 1)
                time.sleep(0.05)
                #x_list.append(x.unsqueeze(3))
                v_list.append(v)
            else:           
                if NoScale:
                    hypara_list = [max(sigma, min_sigma/1)]
                else: 
                    hypara_list = [20.,10.,5.,3.,1.]
                
                ffdnet_num = len(hypara_list)
                tv_num = ffdnet_num
            
                if denoiser=='ffdnet':
                    ffdnet_list = [ffdnet_denosing(x+u, level/255., netvnorm_init).clamp(0, 1) for level in hypara_list]
                if denoiser=='fastdvd':
                    ffdnet_list = [fastdvdnet_denosing(x+u, level/255., netvnorm_init).clamp(0, 1) for level in hypara_list]
                if denoiser=='drunet':
                    ffdnet_list = [DRUNet_Denoise(x+u, level/255., netvnorm_init).clamp(0, 1) for level in hypara_list]
                
                ffdnet_list_list.append(ffdnet_list)
                
                ffdnet_mat = np.stack([x_ele.transpose(1,0).cpu().numpy().reshape(-1).astype(np.float64) for x_ele in ffdnet_list],
                                      axis=0)
                ffdnet_mat_list.append(ffdnet_mat)
        
                
        if  i < 10: 
            v_add = 0
            for idx in range(nChannel):
                v_add += v_list[idx]
            v = v_add/nChannel      
            continue         
        
        if NoMuilt:
            ffdnet_list = ffdnet_list_list[0] # Channel 1
            
            v_add = 0
            for idx in range(ffdnet_num):
                v_add += ffdnet_list[idx]
            v = v_add/ffdnet_num
                    
        else:

            ffdnet_list = ffdnet_list_list[0] # Channel 1
            tv_list = ffdnet_list_list[1]     # Channel 2
            ffdnet_mat = ffdnet_mat_list[0].T
            tv_mat = ffdnet_mat_list[1].T
        
            if NoWeight or NoScale:
                v_add = 0
                for ndx in range(nChannel):
                    tv_list = ffdnet_list_list[ndx]
                    for idx in range(ffdnet_num):
                        v_add += tv_list[idx]
                v = v_add/ffdnet_num/nChannel
            else:
                w_1, w_2 = Alternate_solution(ffdnet_mat[image_n//2:-1:image_m,:], 
                                      tv_mat[image_n//2:-1:image_m,:], iter = 3)
                w_list.append(np.round(w_1,4))
                w_list2.append(np.round(w_2,4))

                v_ffdnet, v_tv = 0, 0
                for idx in range(ffdnet_num):
                    v_ffdnet += w_1[idx] * ffdnet_list[idx]
                for idx in range(tv_num):
                    v_tv += w_2[idx] * tv_list[idx]
                v = 0.5 * (v_ffdnet + v_tv)
        
        
        delta_v = cost(v, v_old)
        v_old = v
        

        if NoScale and delta_v > delta_v_old*0.9 or i-i_old > 10:
            sigma = sigma*0.95
            i_old = i

        if delta_v < 2e-8:
            break # Range
        delta_v_old = delta_v
        
        for iC in range(nChannel):
            #u_list[iC] = u_list[iC] - v + x_list[iC]
            u_list[iC] = 0

    v.clamp_(0, 1)
    im_orig.clamp_(0, 1)

    psnr_ = psnr(v, im_orig)
    #psnr_ = [psnr(x[..., kv], im_orig[..., kv]) for kv in range(image_c)]
    ssim_ = [ssim(v[..., kv], im_orig[..., kv]) for kv in range(image_c)]
    
    return np.mean(psnr_), np.mean(ssim_), v.cpu().numpy(), im_orig.data.cpu().numpy()

def getExtfileList(filepath, extensionName):
    filelists = []
    for eachfilePath, d, filelist in os.walk(filepath):
        for eachfilename in filelist:
            if eachfilename.split(".")[-1].lower() == extensionName.lower():
                tempfile = os.path.join(eachfilePath, eachfilename)
                filelists.append(tempfile)
    return filelists
# In[ ]: 
folder = './data/test/'
file_list = getExtfileList(folder, 'mat')

file_list = [file_list[0]]
# In[ ]: Main
NoMuilt = False#True#
NoScale = False#True#
NoWeight = False#True#

denoiser = 'ffdnet' #{ffdnet, drunet, fastdvd}

w_list = []
w_list2 = []
results = []

file_count = len(file_list)
for index, mat_file in enumerate(file_list):
    print('正在读取第%s/%s个文件:%s' % (index + 1, file_count, mat_file))
    begin_time = time.time()
    
    psnr_res, ssim_res, output, orig= run(mat_file)  # Main Funtion
    result = [mat_file, psnr_res, ssim_res, output, orig]
    results.append(result)
    
    end_time = time.time()
    running_time = end_time - begin_time
    print('Result PSNR {:.2f}, SSIM {:.4f}, Running Time {:.2f}'.format(psnr_res, ssim_res, running_time))

    for kk in range(output.shape[2]):
        ShowData = np.concatenate((orig, output), axis=0)
        plot.imshow(ShowData[:,:,kk])
        plot.show()
