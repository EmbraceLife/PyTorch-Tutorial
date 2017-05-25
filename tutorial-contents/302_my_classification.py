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
python -m pdb tutorial-contents/302_my_classification.py prepareData

## build a net (test)
python -m pdb tutorial-contents/302_my_classification.py build_net


## just display every steps in training (no saving plots)
python -m pdb tutorial-contents/302_my_classification.py train -net /Users/Natsume/Downloads/temp_folders/302/net.pkl -log /Users/Natsume/Downloads/temp_folders/302/log.pkl -p /Users/Natsume/Downloads/temp_folders/302 -d

## to save plots of training
python tutorial-contents/302_my_classification.py train -net /Users/Natsume/Downloads/temp_folders/302noshuffle/net.pkl -log /Users/Natsume/Downloads/temp_folders/302noshuffle/log.pkl -p /Users/Natsume/Downloads/temp_folders/302noshuffle -num 200

## continue to train with full epoch and plots
python tutorial-contents/302_my_classification.py train_again -net /Users/Natsume/Downloads/temp_folders/302noshuffle/net.pkl -log /Users/Natsume/Downloads/temp_folders/302noshuffle/log.pkl -p /Users/Natsume/Downloads/temp_folders/302noshuffle -num 200

## convert images to gif with 3 speeds
python tutorial-contents/302_my_classification.py img2gif -p /Users/Natsume/Downloads/temp_folders/302noshuffle

## all I need to do is to change some key source codes and get a new folder to save plots and net, losses, steps

"""

################################################
# all libraires needed
################################################
import argparse
import sys
import torch
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

	# create tensor (100, 2) of 1s
	n_data = torch.ones(100, 2)

	#### data for class 1
	# torch.normal(mean|tensor, std)
	x0 = torch.normal(2*n_data, 1)
	y0 = torch.zeros(100)

	### data for class 2
	x1 = torch.normal(-2*n_data, 1)
	y1 = torch.ones(100)

	### add two tensor on rows, and change type from int to float
	x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
	y = torch.cat((y0, y1), ).type(torch.LongTensor)

	# conver tensors to variables
	x_v, y_v = Variable(x), Variable(y)

	## plot dataset
	# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
	# plt.show()

	# convert dataset into batches
	torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

	# put whole dataset into batches
	loader = Data.DataLoader(
	    dataset=torch_dataset,      # torch TensorDataset format
		# make smaller batch_size and make shuffle batches, can make loss curve very very wiggy
	    batch_size=100,
	    shuffle=False,               # random shuffle for training
	    num_workers=2,              # subprocesses for loading data
	    drop_last=False				# True: drop smaller remaining data; False: keep smaller remaining data
	)

	return (x_v, y_v, loader)


######################################################
# create Net: network class
######################################################

# move Net in global env due to AttributeError: Can't pickle local object 'build_net.<locals>.Net'
class Net(torch.nn.Module):
	""" 1. create __init__; 2. create forward()
	"""
	def __init__(self, n_feature, n_hidden, n_output):
	    super(Net, self).__init__()
	# 3 lines above are just template must have!!!

		# build 2 hidden layers
	    self.hidden = torch.nn.Linear(n_feature, n_hidden)
	    self.out = torch.nn.Linear(n_hidden, n_output)

	# feed dataset to hidden layers and apply activation functions
	def forward(self, x):
		layer1 = F.relu(self.hidden(x))
		prediction = self.out(layer1)

		return layer1, prediction


######################################################
# build network
######################################################

def build_net(args):
	""" Build network: 1. instantiate a net; 2. build a optimizer box and loss box; 3. build net2pp for printing; 4. return (net, optimizer, loss_func, net2pp)
	"""
	######################
	# hyper parameters:
	learning_rate = 0.02
	optimizer_select = "sgd" # or 'momentum', 'adam', 'rmsprop'
	loss_select = "crossentropy" # or 'mse'

	# input_X has 2 cols;
	# hidden1 has 10 cols? 10 rows?
	# hidden2 has 2 cols? 2 rows?
	# see from many examples
	net = Net(n_feature=2, n_hidden=10, n_output=2)

	######################
	## select an optimizer
	opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=learning_rate)
	opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=learning_rate, momentum=0.8)
	opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=learning_rate, alpha=0.9)
	opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=learning_rate, betas=(0.9, 0.99))
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
	loss_funcs = {'mse':loss_sme, 'crossentropy':loss_crossEntropy}
	loss_func = None
	for k, v in loss_funcs.items():
		if loss_select == k:
			loss_func = v


	# build net2pp for printing (the same object as net)
	net2pp = torch.nn.Sequential(
	    torch.nn.Linear(2, 10),
	    torch.nn.ReLU(),
	    torch.nn.Linear(10, 2)
	)

	return (net, optimizer, loss_func, net2pp)

######################################################
# create plots and save them during training
######################################################
def saveplots(args, param_names, param_values, net2pp):
	""" 1. x, y, plot weights, biases, activations, losses; 2. save plots rather than display
	"""

	####################
	# create figure and outer structure
	####################
	# create figure
	fig = plt.figure(1, figsize=(6, 6))
	# access the current epoch for plotting
	epoch = param_values[-1][0][-1]

	## create figure super title
	# make title net.__repr__() # fontsize="x-large", "large", "medium", "small", or 12
	# remove 'Net (' or 'Sequential (' and '\n)' to print title nicely
	fig.suptitle("epoch:"+str(epoch)+" " + net2pp.__repr__().replace("Sequential (", "").replace("\n)", "").replace("\n", ""), fontsize=8)

	# fig's outer structure's num_row_img, num_col_img
	num_wh_row_col = math.ceil(math.sqrt(len(param_names)))
	# build outer structure
	outer = gridspec.GridSpec(num_wh_row_col, num_wh_row_col)

	# count each layer index
	param_index = 0
	# access each layer (weights, or biases, or activations, or loss)
	for param in param_values:

		# for loss plot
	    if param_names[param_index] == 'loss':
			# define loss plot parameters
			# inner subplot has a single plot
			# num_img_row_col: define how many inner subplots inside an outer subplot
	        num_img_row_col = 1
	        s2 = 1

		# for all other layer plot, if dim == 2
	    elif len(param.size())==2:
			# define layer plot parameters
	        s1, s2 = param.size()
			# if s2 is large, swap values between s1 and s2, make s1 larger
	        if s1 < s2:
	            num_img = s1
	            s1 = s2
	            s2 = num_img
			# num_img_row_col: define how many inner subplots inside an outer subplot
	        num_img_row_col = math.ceil(math.sqrt(s2))

		# for all other layer plot, if dim == 1
	    elif len(param.size()) == 1:
			# define layer plot parameters
			# set subplot num_row_img == 1
			# num_img_row_col: define how many inner subplots inside an outer subplot
	        num_img_row_col = 1
	        s1 = len(param)
	        s2 = 1

	    else:
	        pass

		# all param other than loss must have img_wh, param_padded
		# in order to plot images from arrays
	    if param_names[param_index] != 'loss':
			# inside a outer subplot, get an inner subplot's width and height
	        img_wh = math.ceil(math.sqrt(s1))
			# how many pixel cells are needed to fill with zeros
	        missing_pix = img_wh*img_wh - s1
			# the filled new tensor for plot images
	        param_padded = torch.cat((param.view(s1,s2), torch.zeros((missing_pix, s2))),0)

		# create inner structure: for each outer subplot, create inner structure for a square of inner subplots
	    inner = gridspec.GridSpecFromSubplotSpec(num_img_row_col, num_img_row_col, subplot_spec=outer[param_index], wspace=0.0, hspace=0.0)

		# loop every inner subplots
	    for index in range(s2):
			# get ax for inner subplot ready
	        ax = plt.Subplot(fig, inner[index])

			# plot loss
	        if param_names[param_index] == 'loss':
				# param[0]: list of t or steps
				# param[1]: list of loss
	            ax.plot(param[0], param[1], 'b-')
				# set x-axis and y-axis range
	            ax.set_xlim((0,max(param[0])))
	            ax.set_ylim((0,max(param[1])))
				# set size, color of loss
	            ax.set_title("loss: %.4f" % param[1][-1], fontdict={'size': 8, 'color':  'black'})


	            fig.add_subplot(ax)

			# plot other param or layer
	        else:
				# plot an inner subplot image
	            ax.imshow(np.reshape(param_padded.numpy()[:, index], (img_wh, img_wh)), cmap='gray')

				# If there are more inner subplots, where to put subplot titles
	            if s2 > 1:
	                if index == int(num_img_row_col/2):
	                    ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				# where to subplot title when there is just 1 inner subplot
	            else:
	                ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
	            ax.set_xticks(())
	            ax.set_yticks(())
	            fig.add_subplot(ax)

	    param_index += 1

	fig.savefig('{}/epoch_{}.png'.format(args.plots_path, param[0][-1]))
	# to clear fig for next plotting
	plt.clf()

# just plotting without saving images
def display(args, param_names, param_values, net2pp):
	""" 1. x, y, plot weights, biases, activations, losses; 2. just display the plotting without saving them
	"""

	####################
	# create figure and outer structure
	####################
	# create figure
	fig = plt.figure(1, figsize=(6, 6))
	# access the current epoch for plotting
	epoch = param_values[-1][0][-1]

	## create figure super title
	# make title net.__repr__() # fontsize="x-large", "large", "medium", "small", or 12
	# remove 'Net (' or 'Sequential (' and '\n)' to print title nicely
	fig.suptitle("epoch:"+str(epoch)+" " + net2pp.__repr__().replace("Sequential (", "").replace("\n)", "").replace("\n", ""), fontsize=8)

	# fig's outer structure's num_row_img, num_col_img
	num_wh_row_col = math.ceil(math.sqrt(len(param_names)))
	# build outer structure
	outer = gridspec.GridSpec(num_wh_row_col, num_wh_row_col)

	# count each layer index
	param_index = 0
	# access each layer (weights, or biases, or activations, or loss)
	for param in param_values:

		# for loss plot
	    if param_names[param_index] == 'loss':
			# define loss plot parameters
			# inner subplot has a single plot
			# num_img_row_col: define how many inner subplots inside an outer subplot
	        num_img_row_col = 1
	        s2 = 1

		# for all other layer plot, if dim == 2
	    elif len(param.size())==2:
			# define layer plot parameters
	        s1, s2 = param.size()
			# if s2 is large, swap values between s1 and s2, make s1 larger
	        if s1 < s2:
	            num_img = s1
	            s1 = s2
	            s2 = num_img
			# num_img_row_col: define how many inner subplots inside an outer subplot
	        num_img_row_col = math.ceil(math.sqrt(s2))

		# for all other layer plot, if dim == 1
	    elif len(param.size()) == 1:
			# define layer plot parameters
			# set subplot num_row_img == 1
			# num_img_row_col: define how many inner subplots inside an outer subplot
	        num_img_row_col = 1
	        s1 = len(param)
	        s2 = 1

	    else:
	        pass

		# all param other than loss must have img_wh, param_padded
		# in order to plot images from arrays
	    if param_names[param_index] != 'loss':
			# inside a outer subplot, get an inner subplot's width and height
	        img_wh = math.ceil(math.sqrt(s1))
			# how many pixel cells are needed to fill with zeros
	        missing_pix = img_wh*img_wh - s1
			# the filled new tensor for plot images
	        param_padded = torch.cat((param.view(s1,s2), torch.zeros((missing_pix, s2))),0)

		# create inner structure: for each outer subplot, create inner structure for a square of inner subplots
	    inner = gridspec.GridSpecFromSubplotSpec(num_img_row_col, num_img_row_col, subplot_spec=outer[param_index], wspace=0.0, hspace=0.0)

		# loop every inner subplots
	    for index in range(s2):
			# get ax for inner subplot ready
	        ax = plt.Subplot(fig, inner[index])

			# plot loss
	        if param_names[param_index] == 'loss':
				# param[0]: list of t or steps
				# param[1]: list of loss
	            ax.plot(param[0], param[1], 'b-')
				# set x-axis and y-axis range
	            ax.set_xlim((0,max(param[0])))
	            ax.set_ylim((0,max(param[1])))
				# set size, color of loss
	            ax.set_title("loss: %.4f" % param[1][-1], fontdict={'size': 8, 'color':  'black'})


	            fig.add_subplot(ax)

			# plot other param or layer
	        else:
				# plot an inner subplot image
	            ax.imshow(np.reshape(param_padded.numpy()[:, index], (img_wh, img_wh)), cmap='gray')

				# If there are more inner subplots, where to put subplot titles
	            if s2 > 1:
	                if index == int(num_img_row_col/2):
	                    ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				# where to subplot title when there is just 1 inner subplot
	            else:
	                ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
	            ax.set_xticks(())
	            ax.set_yticks(())
	            fig.add_subplot(ax)

	    param_index += 1

	# control how long to view a fig for each time
	plt.pause(0.5)
	# to clear fig for next plotting
	plt.clf()

def train(args):
	""" Trains a model.
	"""
	# prepare dataset
	x, y, loader = prepareData(args)

	# build net
	net, optimizer, loss_func, net2pp = build_net(args)

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
		for batch_idx, (batch_x, batch_y) in enumerate(loader):

			b_x = Variable(batch_x)
			b_y = Variable(batch_y)

			layer1, prediction = net(b_x)
			loss = loss_func(prediction, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# plots and save every 5 steps or epochs
		# Note: push a tab when save every 5 batches rather than epoch
		if epoch_idx % 5 == 0:

			# every time when plotting, update losses and steps
			losses.append(loss.data.numpy().tolist()[0])
			steps.append(epoch_idx)

			# every time when plotting, update values of x, y, weights, biases, activations, loss
			param_names = []
			param_values = []
			for k, v in net.state_dict().items():
			    param_names.append(k)
			    param_values.append(v)

			param_names.insert(2, "h-layer1")
			param_names.append("pred_layer")
			param_names.insert(0, "y")
			param_names.insert(0, "x")

			param_values.insert(2, layer1.data)
			param_values.append(prediction.data)
			# set y label (classification) from LongTensor to FloatTensor
			# for later operations (inputs must have same type to operate)
			# we are going to plot entire x, y not a single sample
			param_values.insert(0, y.data.type(torch.FloatTensor))
			param_values.insert(0, x.data)

			# losses.append(loss.data[0])
			# steps.append(t)
			param_names.append("loss")
			param_values.append([steps, losses])

			if args.display:
				display(args, param_names, param_values, net2pp)

			else:
				saveplots(args, param_names, param_values, net2pp)

	if args.display:
		plt.ioff()
	else:
		# save net and log
		torch.save((net, net2pp), args.net_path)
		torch.save((steps, losses), args.log_path)
		# convert saved images to gif (speed up, down, normal versions)
		# img2gif(args)

def train_again(args):
	""" Trains a model.
	"""
	# prepare dataset
	x, y, loader = prepareData(args)

	# load net and log
	net, net2pp = torch.load(args.net_path)
	steps, losses = torch.load(args.log_path)

	previous_steps = steps[-1]

	# build workflow
	optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
	loss_func = torch.nn.CrossEntropyLoss()

	# train
	if args.display:
		plt.ion()
	# set t to epoch_idx
	for epoch_idx in range(args.num_epochs):

		# loss value has to be carried in and out
		loss = None
		# batch_idx of an epoch
		for batch_idx, (batch_x, batch_y) in enumerate(loader):

			b_x = Variable(batch_x)
			b_y = Variable(batch_y)

			layer1, prediction = net(b_x)
			loss = loss_func(prediction, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		# plots and save every 5 steps
		if epoch_idx % 5 == 0:
			losses.append(loss.data.numpy().tolist()[0])
			# add up onto previous_steps
			steps.append(previous_steps+epoch_idx)

			param_names = []
			param_values = []
			for k, v in net.state_dict().items():
			    param_names.append(k)
			    param_values.append(v)

			param_names.insert(2, "h-layer1")
			param_names.append("pred_layer")
			param_names.insert(0, "y")
			param_names.insert(0, "x")

			param_values.insert(2, layer1.data)
			param_values.append(prediction.data)
			# set y label (classification) from LongTensor to FloatTensor
			# for later operations (inputs must have same type to operate)
			# we are going to plot entire x, y not a single sample
			param_values.insert(0, y.data.type(torch.FloatTensor))
			param_values.insert(0, x.data)

			# losses.append(loss.data[0])
			# steps.append(t)
			param_names.append("loss")
			param_values.append([steps, losses])

			if args.display:
				display(args, param_names, param_values, net2pp)

			else:
				saveplots(args, param_names, param_values, net2pp)

	if args.display:
		plt.ioff()
	else:
		# save net and log
		torch.save((net, net2pp), args.net_path)
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
	subparser.add_argument('-d', '--display', action='store_true', help='Plot whole process while training')
	subparser.add_argument('-net', '--net_path', required=True, help="Path to save neuralnet model")
	subparser.add_argument('-log', '--log_path', required=True, help="Path to save log information: losses, steps")
	subparser.add_argument('-p', '--plots_path', required=True, help="Path to save plots")
	subparser.add_argument('-num', '--num_epochs', type=int, default=100, help="Number of epochs to train this time")
	subparser.add_argument('-s', '--selection',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')
	subparser.set_defaults(func=train)


#########################################################
	# the command line function defined as train_again
	subparser = subparsers.add_parser('train_again', help='Trains a model.')
	subparser.add_argument('-d', '--display', action='store_true', help='Plot whole process while training')
	subparser.add_argument('-net', '--net_path', required=True, help="Path to load and update neuralnet model")
	subparser.add_argument('-log', '--log_path', required=True, help="Path to load and update log information: losses, steps")
	subparser.add_argument('-p', '--plots_path', required=True, help="Path to save plots")
	subparser.add_argument('-num', '--num_epochs', type=int, default=100,
	                    help="Number of epochs to train this time")
	subparser.set_defaults(func=train_again)


	return parser, subparsers

def img2gif(args):
	os.chdir(args.plots_path)

	# epoch_%d0.png: only takes epoch_0|10|20|30...|100|110
	subprocess.call(['ffmpeg', '-i', 'epoch_%d5.png', 'output.avi'])
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
