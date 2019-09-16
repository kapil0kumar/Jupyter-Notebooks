from torch.utils.data import Dataset
from skimage import io
import torch
import os

from degrade import degrade

class CustomDataset(Dataset):

	def __init__(self, inp_dir, transform=None):

		self.inp_dir = inp_dir
		self.transform = transform

		self.input_images = sorted([os.path.join(inp_dir,i) for i in os.listdir(inp_dir)])

	def __len__(self):

		return len(self.input_images)

	def __getitem__(self, idx):

		output_image = io.imread(self.input_images[idx])
		x=output_image.copy()
		input_image = degrade(x)

		dataset = dict()

		dataset["input"] = input_image
		dataset["output"] = output_image

		if self.transform:
			dataset = self.transform(dataset)

		return dataset

class ToTensor(object):

	def __call__(self, data):
		inp, out = data["input"], data["output"]

		inp = inp.transpose((2,0,1))/255.0
		out = out.transpose((2,0,1))/255.0

		return {
				"input": torch.from_numpy(inp),
				"output": torch.from_numpy(out)
				}



class CustomDatasetTest(Dataset):

	def __init__(self, inp_dir, transform=None):

		self.inp_dir = inp_dir
		self.transform = transform
		self.input_images = sorted([os.path.join(inp_dir,i) for i in os.listdir(inp_dir)])
		self.input_images_name = sorted(os.listdir(inp_dir))

	def __len__(self):
		return len(self.input_images)

	def __getitem__(self, idx):
		image = io.imread(self.input_images[idx])
		image_name = self.input_images_name[idx]

		dataset = dict()

		dataset["image"] = image
		dataset["image_name"] = image_name

		if self.transform:
			dataset = self.transform(dataset)

		return dataset

class ToTensorTest(object):

	def __call__(self,data):
		img = data['image']
		img_name = data['image_name']
		img = img.transpose((2,0,1))/255.0

		return {
				'image': torch.from_numpy(img),
				'image_name': img_name
		}
