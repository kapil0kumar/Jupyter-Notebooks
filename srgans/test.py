import os
import torch
from torch.utils.data import DataLoader
from skimage import io
from torchvision import transforms
import numpy as np

from dataloader_utils import CustomDatasetTest,ToTensorTest
from model import Generator

model = Generator()
model.load_state_dict(torch.load('model_checkpoint/New3_gen_epoch_1.pth'))
model.eval()
if torch.cuda.is_available():
	model=model.cuda()

try:
	os.mkdir("results")
except:
	pass

batch_size= 1

data = CustomDatasetTest(inp_dir='data/degraded', transform=transforms.Compose([ToTensorTest()]))  ### input directory is the directory that contain the degraded images.
dataloader = DataLoader(data, batch_size=batch_size, num_workers=4,shuffle=True)

image_counter= 0

for data in dataloader:
	print("image {}".format(image_counter+1))
	image_counter += 1
	image = data['image']
	image_name = data['image_name'][0]
	orig_image = io.imread('data/test/'+image_name)   # corresponding original image which was in test folder.
	degraded_image=image.numpy()[0].transpose((1,2,0))*255.0
	#io.imsave('results/input{}'.format(image_counter),degraded_image)
	if torch.cuda.is_available():
		image=image.type('torch.FloatTensor').cuda()
	image_out = model(image)
	image_out = image_out.cpu().detach().numpy()[0].transpose((1,2,0))*255.0
	#io.imsave("results/output{}".format(image_counter),image_out)
	comb = np.concatenate((orig_image,degraded_image,image_out),axis=1)   # concatenating images side by side for better comparison.
	io.imsave('results/config2_image{}.jpg'.format(image_counter),comb)
	if image_counter> 25:
		break
