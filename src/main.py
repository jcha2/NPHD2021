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
	def __init__(self, data_dir, suffix):
		sub_dirs = ["negative"+suffix, "positive"+suffix]
		labels = ["0", "1"]
		self.labels = []
		self.filenames = []
		# os.path.join allows only string arguments
		for label in labels:
			path2data = os.path.join(data_dir, sub_dirs[int(label)])
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
	# Unimplemented yet
	# writer = SummaryWriter()

	# Set basic things
	## Set seed for reproducibility
	torch.manual_seed(1004)
	print(torch.__version__)

	# Load dataset
	## It must be modified to take arguments from a user
	histo_dataset = histo_cancer_dataset("../images", "")
	print(len(histo_dataset))
	
	# Divide train and validation dataset
	## TODO: 4, 5, 8, 10 fold CV should be considered in the future
	## Currently, just split into 8:2
	dSize = len(histo_dataset)
	nTrains = int(0.8*dSize)
	nVals = dSize - nTrains
	
	## TODO: Check if random_split preserves 1:1 ratio of positives and negatives
	trains, vals = random_split(histo_dataset, [nTrains, nVals])	

	# Convert images into tensors
	trains = DataLoader(trains, batch_size=32, shuffle=True)
	vals = DataLoader(vals, batch_size=32, shuffle=False)
	
	# Check size
	print(len(trains))
	print(len(vals))

	# Hyperparameters for CoAtNet-7
	## TODO: Which one is the best?
	num_blocks = [2, 2, 3, 5, 2]
	channels = [64, 96, 192, 384, 768]
	block_types = ['C', 'T', 'T', 'T']
	
	## Load CoAtNet
	net = CoAtNet((224, 224), 3, num_blocks, channels, num_classes = 2, block_types = block_types)

	# Loss function
	## CrossEntropyLoss is a single function combining LogSoftMax and NLLLoss (Log Likelihood Loss)
	loss = torch.nn.CrossEntropyLoss()	

	# Update strategy
	optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

	# Notation
	## epoch: the number of times an algorithm visits all dataset
	## iteration: the number of times an algorithm vists a batch
	epoch_=5
	for ep in range(epoch_):
		iter_ = 0
		print(f'epoch: {ep}')
		for x, y in trains:
			y_pred = net(x)
			
			_, y_pred_tags = torch.max(y_pred, dim = 1)
			correct_pred = (y_pred_tags == y).float()
			acc = correct_pred.sum() / len(correct_pred)

			print(f'{acc:.3f}')
			loss_ = loss(y_pred, y)
			print(loss_)
			loss_.backward()
			optimizer.step()
			optimizer.zero_grad()
			iter_ = iter_ + 1
			print(f'iter: {iter_}')
	# Test
	histo_test = histo_cancer_dataset("../images", "_test")
	test_set = DataLoader(histo_test, batch_size=100, shuffle=False)
	for x, y in test_set:
		y_pred = net(x)
		_, y_pred_tags = torch.max(y_pred, dim=1)
		correct_pred = (y_pred_tags == y).float()
		acc = correct_pred.sum() / len(correct_pred)

		print(f'{acc:.3f}')

