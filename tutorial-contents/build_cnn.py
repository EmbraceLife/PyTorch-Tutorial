import argparse
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

## More flexible way to build Net class
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

## Simpler way of build Net class
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

		#### How did 14, 7 are calculated?

	def forward(self, x):
	    conv1 = self.conv1(x)
	    conv2 = self.conv2(conv1)
		# flatten the output of conv2 to (batch_size, 32 * 7 * 7)
	    flat = conv2.view(conv2.size(0), -1)
		# fully connected layer only take input with 2-d
	    logits = self.out(flat)
	    return logits

######################################################
def build_net(args):
	""" Build network: 1. instantiate a net; 2. build a optimizer box and loss box; 3. build net2pp for printing; 4. return (net, optimizer, loss_func, net2pp)
	"""
	######################
	# hyper parameters: can be turned to args.lr, args.opt, args.loss
	learning_rate = 0.02
	optimizer_select = "adam" # or 'momentum', 'adam', 'rmsprop'
	loss_select = "crossentropy" # or 'mse'

	cnn = CNN()
	cnn2pp = CNN2PP()

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

#########################################################
def build_parser():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	# start: description
	parser = argparse.ArgumentParser(description='my argparse tool')
	# create a command line function
	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')
	subparser = subparsers.add_parser('build_net', help='Build networks')
	subparser.set_defaults(func=build_net)

	return parser, subparsers

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

# run the line below in terminal
# python -m pdb tutorial-contents/401_build_nets.py build_net
