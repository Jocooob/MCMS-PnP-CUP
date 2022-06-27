# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:40:47 2021

@author: Administrator
"""

import torch, numpy
import torch.nn as nn
from models import FastDVDnet


model = FastDVDnet(num_input_frames= 5,num_color_channels=1)
state_temp_dict = torch.load('model_gray.pth')
model = model.cuda()
model.load_state_dict(state_temp_dict)
# Sets the model in evaluation mode (e.g. it removes BN)
model.eval()

def fastdvdnet_seqdenoise(seq, noise_std, windsize, model):
	r"""Denoising a video sequence with FastDVDnet.
	
	Parameters 
	----------
	seq : array_like [torch.Tensor]
	      Input noisy video sequence data with size of [N, C, H, W] with 
		  N, C, H, and W being the number of frames, number of color channles 
		  (C=3 for color, C=1 for grayscale), height, and width of the video 
		  sequence to be denoised.
	noise_std : array_like [torch.Tensor]
	      Noise standard deviation with size of [H, W].
	windsize : scalar
		  Temporal window size (number of frames as input to the model).
	model : [torch.nn.Module]
		  Pre-trained model for denoising.
	
	Returns
	-------
	seq_denoised : array_like [torch.Tensor]
		  Output denoised video sequence, with the same size as the input, 
		  that is [Nf, C, H, W].
	"""
	# init arrays to handle contiguous frames and related patches
	# print(seq.shape)
	N, C, H, W = seq.shape
	hw = int((windsize-1)//2) # half window size
	seq_denoised = torch.empty((N, C, H, W)).to(seq.device)
	# input noise map 
	noise_map = noise_std.expand((1, 1, H, W))

	for frameidx in range(N):
		# cicular padding for edge frames in the video sequence
		idx = (torch.tensor(range(frameidx, frameidx+windsize)) - hw)# % N # circular padding Changed!!
		idx = idx.clamp(0, N-1)
		#print(idx)
		noisy_seq = seq[idx].reshape((1, -1, H, W)) # reshape from [W, C, H, W] to [1, W*C, H, W]
		# apply the denoising model to the input datat
		seq_denoised[frameidx] = model(noisy_seq, noise_map)

	return seq_denoised

def fastdvdnet_denoiser(vnoisy, sigma, model=None, useGPU=True, gray=True):
	r"""Denoise an input video (H x W x F x C for color video, and H x W x F for
	     grayscale video) with FastDVDnet
	"""
	# start_time = time.time()
	nColor = 1 if gray else 3 # number of color channels (3 - RGB color, 1 - grayscale)
	# Sets data type according to CPU or GPU modes
	if useGPU:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	if model is None:
		model = FastDVDnet(num_input_frames= 5, num_color_channels=nColor)

		# Load saved weights
		if gray:
			state_temp_dict = torch.load('model_gray.pth') # [pre-trained] model for grayscale videos
		else:
			state_temp_dict = torch.load('model.pth') # [pre-trained] model for color videos
		if useGPU:
			device_ids = [0]
			model = nn.DataParallel(model, device_ids=device_ids).cuda()
		model.load_state_dict(state_temp_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()

	with torch.no_grad():
		# vnoisy = vnoisy.transpose((2,3,0,1)) # [do it in torch] from H x W x F x C to F x C x H x W 
		vnoisy = torch.from_numpy(vnoisy).type('torch.FloatTensor').to(device)
		noisestd = torch.FloatTensor([sigma]).to(device)

		if gray:
			vnoisy = vnoisy.unsqueeze(3) # unsqueeze the color dimension - [H,W,F] to [H,W,F,C=1]
		vnoisy = vnoisy.permute(2, 3, 0, 1) # from H x W x F x C to F x C x H x W 
		# print(vnoisy.finfo, noisestd.finfo)

		# print(torch.max(vnoisy),torch.min(vnoisy))
		# vnoisy = torch.clamp(vnoisy,0.,1.)
		outv = fastdvdnet_seqdenoise( seq=vnoisy,\
									  noise_std=noisestd,\
									  windsize= 5,\
									  model=model )
		# print(outv.shape)
		# print(torch.max(outv),torch.min(outv))
		outv = outv.permute(2, 3, 0, 1) # back from F x C x H x W to H x W x F x C
		if gray:
			outv = outv.squeeze(3) # squeeze the color dimension - [H,W,F,C=1] to [H,W,F]
		outv = outv.data.cpu().numpy()
		# outv = outv.transpose((2,3,0,1)) # [do it in torch] back from F x C x H x W to H x W x F x C
        
	# stop_time = time.time()
	# print('    FastDVDnet video denoising eclipsed in {:.3f}s.'.format(stop_time-start_time))
	return outv


def main(vnoisy, sigma = 0.1):
    vnoisy_array = numpy.asarray(vnoisy)
    denoisyv = fastdvdnet_denoiser(vnoisy_array, sigma, model=model, useGPU=True, gray=True)
    denoisyv_mat = numpy.ascontiguousarray(denoisyv)
    return denoisyv_mat


if __name__ == '__main__': 
    input = numpy.random.rand(256,256,8)
    output = main(input)