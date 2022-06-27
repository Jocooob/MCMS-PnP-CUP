import numpy
import torch, math

from PyDRUNet.networks import UNetRes
n_channels = 1
model = UNetRes(in_nc=n_channels+1, out_nc=n_channels,
                    nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
model_path = './PyDRUNet/model_weight/drunet_gray.pth'
model.load_state_dict(torch.load(model_path), strict=True)
for k, v in model.named_parameters():
    v.requires_grad = False
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
model = model.to(device)

def DRUNet_Denoise(vnoisy, sigma=0.1, netvnorm_init=False):
    #vnoisy = torch.from_numpy(vnoisy).type('torch.FloatTensor').to(device)
    [ww,hh,cc] = vnoisy.size()
    #noise_map = torch.FloatTensor([sigma]).repeat(1, ww, hh).to(device)

    if ww%8!=0:
        vnoisy = torch.cat([vnoisy,vnoisy[ww%8-9:-1,:,:]],0)
    if hh%8!=0:
        vnoisy = torch.cat([vnoisy,vnoisy[:,hh%8-9:-1,:]],1)
        
    if netvnorm_init:
        vnoisy_min = vnoisy.min().item()
        vnoisy_max = vnoisy.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        vnoisy = (vnoisy - vnoisy_min) / (vnoisy_max - vnoisy_min)
        vnoisy = vnoisy * scale + shift

    
    [wt,ht,ct] = vnoisy.size()
    noise_map = torch.FloatTensor([sigma]).repeat(1, wt, ht).to(device)
    vdenoised = torch.empty(wt,ht,ct).to(device)

    with torch.no_grad():
        for idx in range(cc):
            img = vnoisy[:,:,idx].unsqueeze(0)

            img_in = torch.cat([img, noise_map], dim=0)
            img_in = img_in.unsqueeze(0)  # Size is 1*2*W*H

            vdenoised[:,:,idx] = model(img_in).squeeze()
    if netvnorm_init:
        vdenoised = (vdenoised - shift) / scale
        vdenoised = vdenoised * (vnoisy_max - vnoisy_min) + vnoisy_min
    #vdenoised = vdenoised.data.cpu().numpy()
    return vdenoised[0:ww,0:hh,:]


def main(vnoisy, sigma = 0.1):
    vnoisy_array = numpy.asarray(vnoisy)  # matlab array to python array
    
    denoisyv = DRUNet_Denoise(vnoisy_array, sigma)
    
    denoisyv_mat = numpy.ascontiguousarray(denoisyv) # python array to matlab array
    return denoisyv_mat


if __name__ == '__main__': 
    vnoisy = numpy.random.rand(512,512,10)
    output = main(vnoisy)
    
    mse = numpy.mean( (vnoisy-output) ** 2 )
    if mse < 1.0e-10:
        print(100)
    PIXEL_MAX = 1
    psnr=20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print(psnr)
