"""
alias dr pp dir(%1)
alias dt pp %1.__dict__
alias pdt for k, v in %1.items(): print(k, ": ", v)
alias loc locals().keys()
alias doc from inspect import getdoc; from pprint import pprint; pprint(getdoc(%1))
alias sources from inspect import getsourcelines; from pprint import pprint; pprint(getsourcelines(%1))
alias module from inspect import getmodule; from pprint import pprint; pprint(getmodule(%1))
alias fullargs from inspect import getfullargspec; from pprint import pprint; pprint(getfullargspec(%1))

alias opt_param optimizer.param_groups[0]['params'][%1]

alias opt_grad optimizer.param_groups[0]['params'][%1].grad
"""

"""
## View prepared dataset (test prepareData)
python -m pdb tutorial-contents/401_my_cnn.py prepareData -batch_size 500

## build a net (test)
python -m pdb tutorial-contents/401_my_cnn.py build_net

## just train without plot anything or save any plotting
python -m pdb tutorial-contents/401_my_cnn.py train -batch_size 32 -num_batches 50 -num_epochs 10 -num_test 100 -net /Users/Natsume/Downloads/temp_folders/401/net.pkl -log /Users/Natsume/Downloads/temp_folders/401/log.pkl -plot /Users/Natsume/Downloads/temp_folders/401


## just display every steps in training (no saving plots)
python -m pdb tutorial-contents/401_my_cnn.py train -batch_size 1 -num_batches 1 -num_epochs 3 -num_test 100 -net /Users/Natsume/Downloads/temp_folders/401/net.pkl -log /Users/Natsume/Downloads/temp_folders/401/log.pkl -plot /Users/Natsume/Downloads/temp_folders/401 -display -plotting

## to save plots of training
## continue to train for a full epoch 60000 samples and save 3 plots
python -m pdb tutorial-contents/401_my_cnn.py train -batch_size 500 -num_batches 40 -num_epochs 3 -num_test 100 -net /Users/Natsume/Downloads/temp_folders/401/net.pkl -log /Users/Natsume/Downloads/temp_folders/401/log.pkl -plot /Users/Natsume/Downloads/temp_folders/401 -plotting

## continue to train without display or save any plots
python -m pdb tutorial-contents/401_my_cnn.py train_again -batch_size 1 -num_batches 1 -num_epochs 3 -num_test 100 -net /Users/Natsume/Downloads/temp_folders/401/net.pkl -log /Users/Natsume/Downloads/temp_folders/401/log.pkl -plot /Users/Natsume/Downloads/temp_folders/401

## continue to train while just display every steps in training (no saving plots)
python -m pdb tutorial-contents/401_my_cnn.py train_again -batch_size 1 -num_batches 1 -num_epochs 3 -num_test 100 -net /Users/Natsume/Downloads/temp_folders/401/net.pkl -log /Users/Natsume/Downloads/temp_folders/401/log.pkl -plot /Users/Natsume/Downloads/temp_folders/401 -display -plotting

## continue to train for a full epoch 60000 samples and save 3 plots
python -m pdb tutorial-contents/401_my_cnn.py train_again -batch_size 500 -num_batches 40 -num_epochs 3 -num_test 100 -net /Users/Natsume/Downloads/temp_folders/401/net.pkl -log /Users/Natsume/Downloads/temp_folders/401/log.pkl -plot /Users/Natsume/Downloads/temp_folders/401 -plotting

## convert images to gif with 3 speeds
python tutorial-contents/401_my_cnn.py img2gif -p /Users/Natsume/Downloads/temp_folders/401

## all I need to do is to change some key source codes and get a new folder to save plots and net, losses, steps

"""

################################################
# all libraires needed
################################################
import argparse
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
import os
import subprocess

################################################
# prepare data
################################################

def prepareData(args):
	""" Prepare dataset for training later: 1. create x, y dataset; 2. make batches (shuffle, batch_size) 3. return x_v, y_v, loader
	"""
	# reproducible
	torch.manual_seed(1)

	data_path = './mnist/'
	# args.batch_size = 50
	# args.num_batches = 100  # num of batches to train, must < total_num_batches
	download_or_not = False   # set to False if you have downloaded already
	# args.num_test = 100 # number of samples to test from test_data

	# load MNIST dataset into tensors
	train_data = torchvision.datasets.MNIST(
		# where to save the data or load the data from
	    root=data_path,
		# get training dataset only (not test set)
	    train=True,
		# Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
		# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	    transform=torchvision.transforms.ToTensor(),
	    download=download_or_not,
	)
	# it is possible to hack torchvision.datasets.MNIST and torchvision.transforms.ToTensor()????


	# check the original dataset as tensor
	# train_data.train_data.size() # size 60000, 28, 28
	# train_data.train_labels.size() # size 60000
	# train_data.train_data.max() # 255
	# train_data.train_data.__class__ # ByteTensor
	# train_data.train_labels.__class__ # LongTensor
	# train_data.train_labels.max() # 9
	# train_data[0] # return a single tuple of (train_data, train_labels)
	total_num_batches = int(train_data.train_data.__len__()/args.batch_size)

	## plot the first image with gray scale
	# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
	# plt.title('%i' % train_data.train_labels[0])
	# plt.show()

	## use dt train_loader to check its funcs and attributes
	# for index, (img, labels) in enumerate(train_loader): img.size(); break;
	# this way, we see a batch size: (50, 1, 28, 28)
	# train_data must be TensorDataset first already, although its class is torchvision.datasets.mnist.MNIST
	train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
	# once TensorDataset turned into DataLoader, we can only access it by enumerate()

	## extract test dataset
	# without apply transform, dr test_data shows test_data is same type (ByteTensor) and value range 0-255
	test_data = torchvision.datasets.MNIST(root=data_path, train=False)

	## testing data in whole not in batch, we can shrink size to speed up
	# unsqueeze, dim=1: convert (2000, 28, 28) to (2000, 1, 28, 28)
	# shrink to use only first 2000 samples
	# normalize: range 0-1
	# volatile = True: to not calculate gradients
	test_images = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:args.num_test]/255.
	test_labels = test_data.test_labels[:args.num_test]

	return (train_loader, test_images, test_labels)


######################################################
# create Net: network class
######################################################

# build network in flexible way
class CNN(nn.Module):
	""" 1. create __init__; 2. create forward()
	"""
	def __init__(self):
		super(CNN, self).__init__()

		# what exactly is nn.Conv2d???
		self.conv1 = nn.Conv2d(
				in_channels=1,              # input height
				out_channels=16,            # n_filters
				kernel_size=5,              # filter size
				stride=1,                   # filter movement/step
				padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
			)
		self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
		self.out = nn.Linear(32*7*7, 10)

	# feed dataset to hidden layers and apply activation functions
	def forward(self, x):
		conv1_relu = F.relu(self.conv1(x))
		conv1_maxpool = F.max_pool2d(conv1_relu, kernel_size=2)

		conv2_relu = F.relu(self.conv2(conv1_maxpool))
		conv2_maxpool = F.max_pool2d(conv2_relu, kernel_size=2)

		# flatten the output of conv2 to (batch_size, 32 * 7 * 7)
		conv2_flat = conv2_maxpool.view(conv2_maxpool.size(0), -1)
		logits = self.out(conv2_flat)

		return (conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits)




class CNN2PP(nn.Module):
	def __init__(self):
	    super(CNN2PP, self).__init__()

	    self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
			# Note: how to calc cnn output shape
	        nn.Conv2d(
	            in_channels=1,              # input height
	            out_channels=16,            # n_filters
	            kernel_size=5,              # filter size
	            stride=1,                   # filter movement/step
	            padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
	        ),                              # output shape (16, 28, 28)
	        nn.ReLU(),                      # activation
	        nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
	    )
	    self.conv2 = nn.Sequential(         # input shape (1, 28, 28)
	        nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
	        nn.ReLU(),                      # activation
	        nn.MaxPool2d(2),                # output shape (32, 7, 7)
	    )
	    self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

	def forward(self, x):
	    x = self.conv1(x)
	    x = self.conv2(x)
		# flatten the output of conv2 to (batch_size, 32 * 7 * 7)
	    x = x.view(x.size(0), -1)
		# fully connected layer only take input with 2-d
	    output = self.out(x)
	    return output


######################################################
# build network
######################################################

def build_net(args):
	""" Build network: 1. instantiate a net; 2. build a optimizer box and loss box; 3. build net2pp for printing; 4. return (net, optimizer, loss_func, net2pp)
	"""
	######################
	# hyper parameters:
	learning_rate = 0.02
	optimizer_select = "adam" # or 'momentum', 'adam', 'rmsprop'
	loss_select = "crossentropy" # or 'mse'

	######################
	## build instantiate CNN model and CNN2PP to print
	# input_X has 2 cols;
	# hidden1 has 10 cols? 10 rows?
	# hidden2 has 2 cols? 2 rows?
	# see from many examples
	cnn = CNN()
	print(cnn)  # net architecture
	cnn2pp = CNN2PP()
	print(cnn2pp)

	######################
	## select an optimizer
	opt_SGD         = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
	opt_Momentum    = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.8)
	opt_RMSprop     = torch.optim.RMSprop(cnn.parameters(), lr=learning_rate, alpha=0.9)
	opt_Adam        = torch.optim.Adam(cnn.parameters(), lr=learning_rate, betas=(0.9, 0.99))
	optimizers = {'sgd':opt_SGD, 'momentum':opt_Momentum, 'rmsprop':opt_RMSprop, 'adam':opt_Adam}

	# use args.optimizer to select an optimizer to use
	# use args.lr to set learning rate
	optimizer = None
	for k, v in optimizers.items():
		if optimizer_select == k:
			optimizer = v


	######################
	## select a loss_func from 2 possible losses prepared below
	# loss for regression
	loss_mse = torch.nn.MSELoss()
	# CrossEntropyLoss for classification
	loss_crossEntropy = torch.nn.CrossEntropyLoss()
	# put all losses into a dict
	loss_funcs = {'mse':loss_mse, 'crossentropy':loss_crossEntropy}
	loss_func = None
	for k, v in loss_funcs.items():
		if loss_select == k:
			loss_func = v

	return (cnn, optimizer, loss_func, cnn2pp)

######################################################
# create plots and save them during training
######################################################
def saveplots(args, param_names, param_values, cnn):
	""" 1. x, y, plot weights, biases, activations, losses; 2. save plots rather than display
	"""


	####################
	# create figure and outer structure
	####################
	# create figure
	fig = plt.figure(1, figsize=(6, 6))


	#### write suptitle
	# access the current epoch for plotting
	epoch = param_values[-1][0][-1]
	## create figure super title
	# relu and maxpool have no weights, can be ignored to print
	fig.suptitle("epoch:%04d" % epoch + " " + cnn.__repr__().replace("CNN (", "").replace("\n)", "").replace("\n", "").replace("(conv2)", "\n(conv2)").replace("(out)", "\n(out)"), fontsize=8)

	#### outer_frame
	# get outer grid, outer_grid_rows = outer_grid_cols = outer_grid
	outer_grid = math.ceil(math.sqrt(len(param_names)))
	# build outer_frame to hold outer images
	outer_frame = gridspec.GridSpec(outer_grid, outer_grid)

	#### loop through each outer images
	param_index = 0
	# each outer_image is a param_values: input_image, w, b, activations, loss
	for param in param_values:

		## build inner image for loss
		if param_names[param_index] == 'loss':
			# inner_grid for loss
			inner_grid_loss = 1
			num_inner_img = 1
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid_loss, inner_grid_loss, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])
				ax.plot(param[0], param[1], 'b-')
				# set x-axis and y-axis range
				ax.set_xlim((0,max(param[0])))
				ax.set_ylim((0,max(param[1])))
				# set size, color of loss
				ax.set_title("loss: %.4f" % param[1][-1], fontdict={'size': 8, 'color':  'black'})
				fig.add_subplot(ax)

		## build inner image for input_image
		elif param_names[param_index] == 'image':
			# inner_grid for loss
			inner_grid_inputImage = 1
			num_inner_img = 1
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid_inputImage, inner_grid_inputImage, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])
				ax.imshow(param[0], cmap='gray')

				# set size, color of input image
				ax.set_title(param_names[param_index]+": {}".format(param[0].shape), fontdict={'size': 8, 'color':  'black'})
				ax.set_xticks(())
				ax.set_yticks(())
				fig.add_subplot(ax)

		## build inner image for biases or activations with just 1-d
		elif len(param.size()) == 1:

			inner_grid = 1
			num_inner_img = 1
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

			inner_img_width = math.ceil(math.sqrt(len(param)))
			# how many pixel cells are needed to fill with zeros
			missing_pix = inner_img_width*inner_img_width - len(param)
			# the filled new tensor for plot images
			param_padded = torch.cat((param.view(len(param),1), torch.zeros((missing_pix, 1))),0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])

				ax.imshow(param_padded.view(inner_img_width, inner_img_width).numpy(), cmap='gray')

				# set size, color of input image
				ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				ax.set_xticks(())
				ax.set_yticks(())
				fig.add_subplot(ax)


		## build inner image for biases or activations with just 1-d
		elif len(param.size()) == 2:
			# define layer plot parameters
			s1, s2 = param.size()
			# if s2 is large, swap values between s1 and s2, make s1 larger
			if s1 < s2:
				num_img = s1
				s1 = s2
				s2 = num_img
			# num_img_row_col: define how many inner subplots inside an outer subplot
			inner_grid = math.ceil(math.sqrt(s2))
			num_inner_img = s2
			inner_img_width = math.ceil(math.sqrt(s1))
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

			# how many pixel cells are needed to fill with zeros
			missing_pix = inner_img_width*inner_img_width - s1
			# the filled new tensor for plot images
			param_padded = torch.cat((param.view(s1, s2), torch.zeros((missing_pix, s2))),0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])

				ax.imshow(param_padded.numpy()[:, sub_sub].reshape(inner_img_width, inner_img_width), cmap='gray')

				if sub_sub == inner_grid-2:
					ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				ax.set_xticks(())
				ax.set_yticks(())
				fig.add_subplot(ax)

		## build inner image for biases or activations with just 1-d
		elif len(param.size()) >= 3:
			param = torch.squeeze(param)

			if len(param.size()) == 3:

				s2, inner_img_width, _ = param.size()
				inner_grid = math.ceil(math.sqrt(s2))
				num_inner_img = s2
				inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

				for sub_sub in range(num_inner_img):
					ax = plt.Subplot(fig, inner_frame[sub_sub])

					ax.imshow(param.numpy()[sub_sub], cmap='gray')

					if sub_sub == inner_grid-2:
						ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
					ax.set_xticks(())
					ax.set_yticks(())
					fig.add_subplot(ax)

			else:
				s2, s3, inner_img_width, _ = param.size()
				inner_grid = math.ceil(math.sqrt(s2))
				deep_grid = math.ceil(math.sqrt(s3))
				num_inner_img = s2
				num_deep_img = s3
				inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

				for sub_sub in range(num_inner_img):

					deep_frame = gridspec.GridSpecFromSubplotSpec(deep_grid, deep_grid, subplot_spec=inner_frame[sub_sub], wspace=0.0, hspace=0.0)

					for deep in range(num_deep_img):

						ax = plt.Subplot(fig, deep_frame[deep])
						ax.imshow(param[sub_sub, deep, :, :].numpy(), cmap='gray')

						if sub_sub == inner_grid-2 and deep == inner_grid-2:
							ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
						ax.set_xticks(())
						ax.set_yticks(())
						fig.add_subplot(ax)
#############################
		param_index += 1

	fig.savefig('{}/epoch_{}.png'.format(args.plots_path, epoch))
	# to clear fig for next plotting
	plt.clf()

# just plotting without saving images
def display(args, param_names, param_values, cnn):
	""" 1. x, y, plot weights, biases, activations, losses; 2. just display the plotting without saving them
	"""

	####################
	# create figure and outer structure
	####################
	# create figure
	fig = plt.figure(1, figsize=(6, 6))


	#### write suptitle
	# access the current epoch for plotting
	epoch = param_values[-1][0][-1]
	## create figure super title
	# relu and maxpool have no weights, can be ignored to print
	fig.suptitle("epoch:%04d" % epoch + " " + cnn.__repr__().replace("CNN (", "").replace("\n)", "").replace("\n", "").replace("(conv2)", "\n(conv2)").replace("(out)", "\n(out)"), fontsize=8)

	#### outer_frame
	# get outer grid, outer_grid_rows = outer_grid_cols = outer_grid
	outer_grid = math.ceil(math.sqrt(len(param_names)))
	# build outer_frame to hold outer images
	outer_frame = gridspec.GridSpec(outer_grid, outer_grid)

	#### loop through each outer images
	param_index = 0
	# each outer_image is a param_values: input_image, w, b, activations, loss
	for param in param_values:

		## build inner image for loss
		if param_names[param_index] == 'loss':
			# inner_grid for loss
			inner_grid_loss = 1
			num_inner_img = 1
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid_loss, inner_grid_loss, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])
				ax.plot(param[0], param[1], 'b-')
				# set x-axis and y-axis range
				ax.set_xlim((0,max(param[0])))
				ax.set_ylim((0,max(param[1])))
				# set size, color of loss
				ax.set_title("loss: %.4f" % param[1][-1], fontdict={'size': 8, 'color':  'black'})
				fig.add_subplot(ax)

		## build inner image for input_image
		elif param_names[param_index] == 'image':
			# inner_grid for loss
			inner_grid_inputImage = 1
			num_inner_img = 1
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid_inputImage, inner_grid_inputImage, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])
				ax.imshow(param[0], cmap='gray')

				# set size, color of input image
				ax.set_title(param_names[param_index]+": {}".format(param[0].shape), fontdict={'size': 8, 'color':  'black'})
				ax.set_xticks(())
				ax.set_yticks(())
				fig.add_subplot(ax)

		## build inner image for biases or activations with just 1-d
		elif len(param.size()) == 1:

			inner_grid = 1
			num_inner_img = 1
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

			inner_img_width = math.ceil(math.sqrt(len(param)))
			# how many pixel cells are needed to fill with zeros
			missing_pix = inner_img_width*inner_img_width - len(param)
			# the filled new tensor for plot images
			param_padded = torch.cat((param.view(len(param),1), torch.zeros((missing_pix, 1))),0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])

				ax.imshow(param_padded.view(inner_img_width, inner_img_width).numpy(), cmap='gray')

				# set size, color of input image
				ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				ax.set_xticks(())
				ax.set_yticks(())
				fig.add_subplot(ax)


		## build inner image for biases or activations with just 2-d
		elif len(param.size()) == 2:
			# define layer plot parameters
			s1, s2 = param.size()
			# if s2 is large, swap values between s1 and s2, make s1 larger
			if s1 < s2:
				num_img = s1
				s1 = s2
				s2 = num_img
			# num_img_row_col: define how many inner subplots inside an outer subplot
			inner_grid = math.ceil(math.sqrt(s2))
			num_inner_img = s2
			inner_img_width = math.ceil(math.sqrt(s1))
			inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

			# how many pixel cells are needed to fill with zeros
			missing_pix = inner_img_width*inner_img_width - s1
			# the filled new tensor for plot images
			param_padded = torch.cat((param.view(s1, s2), torch.zeros((missing_pix, s2))),0)
			## plot loss
			for sub_sub in range(num_inner_img):
				ax = plt.Subplot(fig, inner_frame[sub_sub])

				ax.imshow(param_padded.numpy()[:, sub_sub].reshape(inner_img_width, inner_img_width), cmap='gray')

				if sub_sub == inner_grid-2:
					ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				ax.set_xticks(())
				ax.set_yticks(())
				fig.add_subplot(ax)

		## build inner image for biases or activations with just 3-d or 4d
		elif len(param.size()) >= 3:
			param = torch.squeeze(param)

			# size 3d
			if len(param.size()) == 3:

				s2, inner_img_width, _ = param.size()
				inner_grid = math.ceil(math.sqrt(s2))
				num_inner_img = s2
				inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

				for sub_sub in range(num_inner_img):
					ax = plt.Subplot(fig, inner_frame[sub_sub])

					ax.imshow(param.numpy()[sub_sub], cmap='gray')

					if sub_sub == inner_grid-2:
						ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
					ax.set_xticks(())
					ax.set_yticks(())
					fig.add_subplot(ax)

			# size: 4d
			else:
				s2, s3, inner_img_width, _ = param.size()
				inner_grid = math.ceil(math.sqrt(s2))
				deep_grid = math.ceil(math.sqrt(s3))
				num_inner_img = s2
				num_deep_img = s3
				inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[param_index], wspace=0.0, hspace=0.0)

				for sub_sub in range(num_inner_img):

					deep_frame = gridspec.GridSpecFromSubplotSpec(deep_grid, deep_grid, subplot_spec=inner_frame[sub_sub], wspace=0.0, hspace=0.0)

					for deep in range(num_deep_img):

						ax = plt.Subplot(fig, deep_frame[deep])
						ax.imshow(param[sub_sub, deep, :, :].numpy(), cmap='gray')

						if sub_sub == inner_grid-2 and deep == inner_grid-2:
							ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
						ax.set_xticks(())
						ax.set_yticks(())
						fig.add_subplot(ax)
#############################

		param_index += 1

	# control how long to view a fig for each time
	plt.pause(0.5)
	# to clear fig for next plotting
	plt.clf()

def train(args):
	""" Trains a model.
	"""
	# prepare dataset
	train_loader, test_images, test_labels = prepareData(args)

	# build net
	cnn, optimizer, loss_func, cnn2pp = build_net(args)

	# train
	losses = []
	steps = []

	if args.display:
		plt.ion()

	# for every epoch of training
	for epoch_idx in range(args.num_epochs):

		# loss value has to be carried in and out
		loss = None

		# traing model for every batch
		for batch_idx, (batch_img, batch_lab) in enumerate(train_loader):

			b_img = Variable(batch_img)
			b_lab = Variable(batch_lab)

			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits = cnn(b_img)
			loss = loss_func(logits, b_lab)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print("loss for batch_{}: %.4f".format(batch_idx) % loss.data.numpy()[0])
			# don't train the full epoch or total_num_batches, but only specific num_batches in each epoch
			if args.num_batches == batch_idx:
				break


		# plot every n epochs
		if args.plotting == True and epoch_idx % 1 == 0:

			# keep test and plot based on a single and same image
			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits = cnn(torch.unsqueeze(test_images[0], dim=0))
			# Note: input_image has to be 4d tensor (n, 1, 28, 28)

			# every time when plotting, update losses and steps
			losses.append(loss.data.numpy().tolist()[0])
			steps.append(epoch_idx+1)

			# every time when plotting, update values of x, y, weights, biases, activations, loss
			param_names = []
			param_values = []
			for k, v in cnn.state_dict().items():
			    param_names.append(k)
			    param_values.append(v)

			## insert conv2_maxpool and conv2_relu
			param_names.insert(4, "2maxPool")
			param_names.insert(4, "2relu")
			param_values.insert(4, conv2_maxpool.data[0])
			param_values.insert(4, conv2_relu.data[0])

			## insert conv1_maxpool and conv1_relu
			param_names.insert(2, "1maxPool")
			param_names.insert(2, "1relu")
			param_values.insert(2, conv1_maxpool.data[0])
			param_values.insert(2, conv1_relu.data[0])

			## insert a single image and its label
			param_names.insert(0, "image")
			test_img1 = test_images.data.numpy()[0] # (1, 28, 28)
			np_img1 = np.squeeze(test_img1) # (28, 28)
			test_lab1 = test_labels[0]
			# insert a single image and label for plotting loop
			param_values.insert(0, (np_img1, test_lab1))


			## append logits for a single images
			logits1 = logits[0]
			logits1_softmax = F.softmax(logits1).data
			param_names.append("softmax")
			param_values.append(logits1_softmax)

			## append losses and steps
			# losses.append(loss.data[0])
			# steps.append(t)
			param_names.append("loss")
			param_values.append([steps, losses])
			# check size of all layers except image and loss
			# pp [p.size() for p in param_values[1:-1]]

			# shorten param_names
			shorten_names = [p_name.replace("weight", "w").replace("bias", "b") for p_name in param_names]
			param_names = shorten_names

			if args.display:
				display(args, param_names, param_values, cnn)

			else:
				saveplots(args, param_names, param_values, cnn)

			# epoch counting (start 1 not 0)
			print("finish saving plot for epoch_%d" % epoch_idx+1)

	if args.display:
		plt.ioff()
	else:
		# save net and log
		torch.save(cnn, args.net_path)
		torch.save((steps, losses), args.log_path)
		# convert saved images to gif (speed up, down, normal versions)
		# img2gif(args)

def train_again(args):
	""" Trains a model.
	"""

	# prepare dataset
	train_loader, test_images, test_labels = prepareData(args)

	# load net and log
	cnn = torch.load(args.net_path)

	steps, losses = torch.load(args.log_path)
	previous_steps = steps[-1]

	# build workflow
	optimizer = torch.optim.Adam(cnn.parameters(), lr=0.02, betas=(0.9, 0.99))
	loss_func = torch.nn.CrossEntropyLoss()

		# train

	if args.display:
		plt.ion()

	# for every epoch of training
	for epoch_idx in range(args.num_epochs):

		# loss value has to be carried in and out
		loss = None

		# traing model for every batch
		for batch_idx, (batch_img, batch_lab) in enumerate(train_loader):

			b_img = Variable(batch_img)
			b_lab = Variable(batch_lab)

			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits = cnn(b_img)
			loss = loss_func(logits, b_lab)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Two ways to format print strings
			print("loss for batch_{}: %.4f".format(batch_idx) % loss.data.numpy()[0])
			# print("loss for batch_%d: %.4f" % (batch_idx, loss.data.numpy()[0]))
			# don't train the full epoch or total_num_batches, but only specific num_batches in each epoch
			if args.num_batches == batch_idx:
				break

		# plot every n epochs
		if args.plotting == True and epoch_idx % 1 == 0:

			# keep test and plot based on a single and same image
			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits = cnn(torch.unsqueeze(test_images[0], dim=0))
			# Note: input_image has to be 4d tensor (n, 1, 28, 28)

			# every time when plotting, update losses and steps
			losses.append(loss.data.numpy().tolist()[0])
			steps.append(previous_steps+epoch_idx+1)

			# every time when plotting, update values of x, y, weights, biases, activations, loss
			param_names = []
			param_values = []
			for k, v in cnn.state_dict().items():
			    param_names.append(k)
			    param_values.append(v)

			## insert conv2_maxpool and conv2_relu
			param_names.insert(4, "2maxPool")
			param_names.insert(4, "2relu")
			param_values.insert(4, conv2_maxpool.data[0])
			param_values.insert(4, conv2_relu.data[0])

			## insert conv1_maxpool and conv1_relu
			param_names.insert(2, "1maxPool")
			param_names.insert(2, "1relu")
			param_values.insert(2, conv1_maxpool.data[0])
			param_values.insert(2, conv1_relu.data[0])

			## insert a single image and its label
			param_names.insert(0, "image")
			test_img1 = test_images.data.numpy()[0] # (1, 28, 28)
			np_img1 = np.squeeze(test_img1) # (28, 28)
			test_lab1 = test_labels[0]
			# insert a single image and label for plotting loop
			param_values.insert(0, (np_img1, test_lab1))


			## append logits for a single images
			logits1 = logits[0]
			logits1_softmax = F.softmax(logits1).data
			param_names.append("softmax")
			param_values.append(logits1_softmax)

			## append losses and steps
			# losses.append(loss.data[0])
			# steps.append(t)
			param_names.append("loss")
			param_values.append([steps, losses])
			# check size of all layers except image and loss
			# pp [p.size() for p in param_values[1:-1]]

			# shorten param_names
			shorten_names = [p_name.replace("weight", "w").replace("bias", "b") for p_name in param_names]
			param_names = shorten_names

			if args.display:
				display(args, param_names, param_values, cnn)

			else:
				saveplots(args, param_names, param_values, cnn)

	if args.display:
		plt.ioff()
	else:
		# save net and log
		torch.save(cnn, args.net_path)
		torch.save((steps, losses), args.log_path)
		# convert saved images to gif (speed up, down, normal versions)
		# img2gif(args)

def build_parser():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	# start: description
	parser = argparse.ArgumentParser(description='my argparse tool')

	# create a command line function
	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	#########################################################
	subparser = subparsers.add_parser('prepareData', help='Preprocess dataset for training')
	subparser.add_argument('-batch_size', type=int, default=50, help="Number of samples in each batch")
	subparser.set_defaults(func=prepareData)

	#########################################################
	subparser = subparsers.add_parser('build_net', help='Build network')
	subparser.set_defaults(func=build_net)

	#########################################################
	subparser = subparsers.add_parser('img2gif', help='conver images to gif with 3 speeds')
	subparser.add_argument('-p', '--plots_path', required=True, help="Path to save plots")
	subparser.set_defaults(func=img2gif)

	#########################################################
	subparser = subparsers.add_parser('train', help='Trains a model for the first time.')
	# add args to train function
	subparser.add_argument('-plotting', action='store_true', help='either display or save plotting; with display being true, just display images; if display is false, save plots')
	subparser.add_argument('-batch_size', type=int, default=50, help="Number of samples in each batch")
	subparser.add_argument('-num_batches', type=int, default=100, help="Number of batches to train in each epoch")
	subparser.add_argument('-num_test', type=int, default=1000, help="Number of samples to test during testing")
	subparser.add_argument('-display', action='store_true', help='Plot whole process while training')
	subparser.add_argument('-net', '--net_path', required=True, help="Path to save neuralnet model")
	subparser.add_argument('-log', '--log_path', required=True, help="Path to save log information: losses, steps")
	subparser.add_argument('-plot', '--plots_path', required=True, help="Path to save plots")
	subparser.add_argument('-num_epochs', type=int, default=1, help="Number of epochs to train this time")
	subparser.add_argument('-s', '--selection',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')
	subparser.set_defaults(func=train)


#########################################################
	# the command line function defined as train_again
	subparser = subparsers.add_parser('train_again', help='Trains a model.')
	# add args to train function
	subparser.add_argument('-plotting', action='store_true', help='either display or save plotting; with display being true, just display images; if display is false, save plots')
	subparser.add_argument('-batch_size', type=int, default=50, help="Number of samples in each batch")
	subparser.add_argument('-num_batches', type=int, default=100, help="Number of batches to train in each epoch")
	subparser.add_argument('-num_test', type=int, default=1000, help="Number of samples to test during testing")
	subparser.add_argument('-display', action='store_true', help='Plot whole process while training')
	subparser.add_argument('-net', '--net_path', required=True, help="Path to save neuralnet model")
	subparser.add_argument('-log', '--log_path', required=True, help="Path to save log information: losses, steps")
	subparser.add_argument('-plot', '--plots_path', required=True, help="Path to save plots")
	subparser.add_argument('-num_epochs', type=int, default=1, help="Number of epochs to train this time")
	subparser.add_argument('-s', '--selection',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')
	subparser.set_defaults(func=train_again)


	return parser, subparsers

def img2gif(args):
	os.chdir(args.plots_path)

	# epoch_%d0.png: only takes epoch_0|10|20|30...|100|110
	# epoch_%d.png: 1,2,3,4,5 plot every epoch
	subprocess.call(['ffmpeg', '-i', 'epoch_%d.png', 'output.avi'])
	subprocess.call(['ffmpeg', '-i', 'output.avi', '-filter:v', 'setpts=4.0*PTS', 'output_down.avi'])
	subprocess.call(['ffmpeg', '-i', 'output.avi', '-r', '16', '-filter:v', 'setpts=0.25*PTS', 'output_up.avi'])
	subprocess.call(['ffmpeg', '-i', 'output_down.avi', 'out_down.gif'])
	subprocess.call(['ffmpeg', '-i', 'output_up.avi', 'out_up.gif'])
	subprocess.call(['ffmpeg', '-i', 'output.avi', 'out.gif'])


def parse_args(parser):
	""" Parses command-line arguments.
	"""
	return parser.parse_args()

def main():
	parser, _ = build_parser()
	args = parse_args(parser)

	sys.exit(args.func(args) or 0)

if __name__ == '__main__':
	main()
