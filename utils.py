import torch
import torchvision.utils as vutils
import math
import os
from torch import autograd

def generate_imgs(img, n_dms, gen, samples_path, step=0, is_cuda=True):
	gen.eval()
	m = img.shape[0]

	lbl = torch.arange(start=-1, end=n_dms)
	lbl = lbl.expand(m, n_dms+1).reshape([-1])
	
	if is_cuda:
		lbl = lbl.cuda()
	img_ = torch.repeat_interleave(img, n_dms+1, dim=0)

	real_idx = torch.arange(start=0, end=m*(n_dms+1), step=n_dms+1)   
	lbl[real_idx] = 0

	display_imgs = gen(img_, lbl)
	display_imgs[real_idx] = img
	
	display_imgs_ = vutils.make_grid(display_imgs, normalize=True, nrow=n_dms+1, padding=2, pad_value=1)
	vutils.save_image(display_imgs_, os.path.join(samples_path, 'sample_' + str(step) + '.png'))

def gradient_penalty(real, fake, critic, is_cuda=True):
	m = real.shape[0]
	epsilon = torch.rand(m, 1, 1, 1)
	if is_cuda:
		epsilon = epsilon.cuda()
	
	interpolated_img = epsilon * real + (1-epsilon) * fake
	interpolated_out, _ = critic(interpolated_img)

	grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
							   grad_outputs=torch.ones(interpolated_out.shape).cuda() if is_cuda else torch.ones(interpolated_out.shape),
							   create_graph=True, retain_graph=True)[0]
	grads = grads.reshape([m, -1])
	grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() 
	return grad_penalty