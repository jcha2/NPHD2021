from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from coatnet import CoAtNet
import coatnet

from torch.utils.tensorboard import SummaryWriter

# Set static random seed for reproducibility
def seed_everything(seed_value):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	os.environ['PYTHONHASHSEED'] = str(seed_value)
    
	if torch.cuda.is_available(): 
		torch.cuda.manual_seed(seed_value)
		torch.cuda.manual_seed_all(seed_value)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True

# What is argument for the class
# Inheritance "Dataset" Class
class histo_cancer_dataset(Dataset):
	def __init__(self, data_dir):
		labels = ["0", "1"]
		self.labels = []
		self.filenames = []
		# os.path.join allows only string arguments
		for label in labels:
			path2data = os.path.join(data_dir, label)
			filenames = os.listdir(path2data)
			self.filenames = self.filenames+[os.path.join(path2data, f) for f in filenames]
			self.labels = self.labels + [int(label)] * len(filenames)
		
		# convert to tesnsor type
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		image = Image.open(self.filenames[idx])
		image = self.transform(image)
		return image, self.labels[idx]
	
if __name__ == '__main__':
	# load tensorboard for pytorch
	writer = SummaryWriter()
	torch.manual_seed(1004)
	print(torch.__version__)

	histo_dataset = histo_cancer_dataset("./kTest/")
	print(len(histo_dataset))
	
	dSize = len(histo_dataset)
	nTrains = int(0.8*dSize)
	nVals = dSize - nTrains

	trains, vals = random_split(histo_dataset, [nTrains, nVals])	

	trains = DataLoader(trains, batch_size=32, shuffle=True)
	vals = DataLoader(vals, batch_size=32, shuffle=False)

	print(len(trains))
	print(len(vals))
	
	for x, y in trains:
		print(x.shape)
		print(y.shape)
		break

	num_blocks = [2, 2, 3, 5, 2]
	channels = [64, 96, 192, 384, 768]
	block_types = ['C', 'T', 'T', 'T']

	net = CoAtNet((224, 224), 3, num_blocks, channels, num_classes = 2, block_types = block_types)

	criterion = torch.nn.Softmax(dim=1)
	optimizer = torch.optim.Adam(net.parameters(), lr = 0.0000001)
	loss = torch.nn.CrossEntropyLoss()

	iter_=0
	epoch_=5
	for ep in range(epoch_):
		print(f'epoch: {ep}')
		for x, y in trains:
			out = net(x)
			y_pred = criterion(out)

			_, y_pred_tags = torch.max(y_pred, dim = 1)
			correct_pred = (y_pred_tags == y).float()
			acc = correct_pred.sum() / len(correct_pred)

			print(f'{acc:.3f}')
			loss_ = loss(out, y)
			print(loss_)
			loss_.backward()
			optimizer.step()
			optimizer.zero_grad()
			iter_ = iter_ + 1
			print(iter_)
#	print(out.shape, coatnet.count_parameters(net))

