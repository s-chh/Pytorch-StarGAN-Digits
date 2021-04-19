from model import Discriminator, Generator
from data_loader import get_loader
from utils import generate_imgs, gradient_penalty
from torch import optim
import torch
import os
import torch.nn as nn
from torchsummary import summary
import pdb
import random
import numpy as np

EPOCHS = 100  # 50-300
BATCH_SIZE = 128
IMGS_TO_DISPLAY = 25
LOAD_MODEL = False

IMAGE_SIZE = 32
NUM_DOMAINS = 5

N_CRITIC = 5
GRADIENT_PENALTY = 10 

# Directories for storing model and output samples
model_path = './model'
os.makedirs(model_path, exist_ok=True)
samples_path = './samples'
os.makedirs(samples_path, exist_ok=True)
db_path = './data'
os.makedirs(samples_path, exist_ok=True)

# Networks
gen = Generator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)
dis = Discriminator(num_domains=NUM_DOMAINS, image_size=IMAGE_SIZE)

print(gen)
summary(gen, (3, IMAGE_SIZE, IMAGE_SIZE), device='cpu')
print(dis)
summary(dis, (3, IMAGE_SIZE, IMAGE_SIZE), device='cpu')

# Load previous model   
if LOAD_MODEL:
	gen.load_state_dict(torch.load(os.path.join(model_path, 'gen.pkl')))
	dis.load_state_dict(torch.load(os.path.join(model_path, 'dis.pkl')))

# Define Optimizers
g_opt = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
d_opt = optim.Adam(dis.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Define Loss
ce = nn.CrossEntropyLoss()

# Data loaders
ds_loader = get_loader(db_path, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
iters_per_epoch = len(ds_loader)

# Fix images for viz
img_fixed = iter(ds_loader).next()[0][:IMGS_TO_DISPLAY]

# GPU Compatibility
is_cuda = torch.cuda.is_available()
if is_cuda:
	gen, dis = gen.cuda(), dis.cuda()
	img_fixed = img_fixed.cuda()

# Cycle-GAN Training
total_iter = 0
g_gan_loss = g_clf_loss = g_rec_loss = torch.Tensor([0])
for epoch in range(EPOCHS):
	gen.train()
	dis.train()

	for i, data in enumerate(ds_loader):
		total_iter += 1

		# Loading data
		real, dm = data
		
		if is_cuda:
			real, dm = real.cuda(), dm.cuda()

		target_dm = dm[torch.randperm(dm.size(0))]

		# Fake Images
		fake = gen(real, target_dm)

		# Training discriminator
		real_gan_out, real_cls_out = dis(real)
		fake_gan_out, fake_cls_out = dis(fake.detach())

		d_gan_loss = -(real_gan_out.mean() - fake_gan_out.mean()) + gradient_penalty(real, fake, dis, is_cuda) * GRADIENT_PENALTY
		d_clf_loss = ce(real_cls_out, dm)

		d_opt.zero_grad()
		d_loss = d_gan_loss + d_clf_loss
		d_loss.backward()
		d_opt.step()

		# Training Generator
		if total_iter % N_CRITIC == 0:
			fake = gen(real, target_dm)
			fake_gan_out, fake_cls_out = dis(fake)

			g_gan_loss = - fake_gan_out.mean()
			g_clf_loss = ce(fake_cls_out, target_dm)
			g_rec_loss = (real - gen(fake, dm)).abs().mean()

			g_opt.zero_grad()
			g_loss = g_gan_loss + g_clf_loss + g_rec_loss
			g_loss.backward()
			g_opt.step()

		if i % 50 == 0:
			print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS)
				  + " iter: " + str(i+1) + "/" + str(iters_per_epoch)
				  + " total_iters: " + str(total_iter)
				  + "\td_gan_loss:" + str(round(d_gan_loss.item(), 4))
				  + "\td_clf_loss:" + str(round(d_clf_loss.item(), 4))
				  + "\tg_gan_loss:" + str(round(g_gan_loss.item(), 4))
				  + "\tg_clf_loss:" + str(round(g_clf_loss.item(), 4))
				  + "\tg_rec_loss:" + str(round(g_rec_loss.item(), 4)))

		if total_iter % 1000==0:
			generate_imgs(img_fixed, NUM_DOMAINS, gen, samples_path, total_iter, is_cuda)

	torch.save(gen.state_dict(), os.path.join(model_path, 'gen.pkl'))
	torch.save(dis.state_dict(), os.path.join(model_path, 'dis.pkl'))
	
generate_imgs(img_fixed, NUM_DOMAINS, gen, samples_path, is_cuda=is_cuda)
