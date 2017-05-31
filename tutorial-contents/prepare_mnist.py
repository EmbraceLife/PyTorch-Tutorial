import argparse
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

################################################
# prepare data
################################################

def prepareData(args):
	""" Prepare dataset for training later: 1. create x, y dataset; 2. make batches (shuffle, batch_size) 3. return x_v, y_v, loader
	"""

	###################################
	## hyper-parameters
	torch.manual_seed(1) # reproducible
	data_path = args.mnist_dir # dir to store dataset
	download_or_not = True # if not available, then download; if available it is fine
	do_plot = False # plot a single sample of image or not

	# load MNIST dataset into tensors
	train_data = torchvision.datasets.MNIST(
	    root=data_path,
	    train=True, # training set
	    transform=torchvision.transforms.ToTensor(),
	    download=download_or_not,
	)

	###################################
	## explore
	"""
	# how many datasets are available?
	dr torchvision.datasets
	# How to add my own dataset?
	sources torchvision.datasets.MNIST
	sources torchvision.datasets
	# explore train_data
	dr train_data
	"""

	###################################
	## plot the first image with gray scale
	if do_plot:
		plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
		plt.title('%i' % train_data.train_labels[0])
		plt.show()

	###################################
	# training set turned into randomly shuffled batches
	train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

	###################################
	# batch dataset is accessed through enumerate()
	for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
		print("batch_index: %d" % batch_idx,
				"\n batch_images: type: %s" % batch_images.__class__,
				"\t size: {}".format(batch_images.size()),
				"\n batch_labels: type: %s" % batch_labels.__class__,
				"\n size: {}".format(batch_labels.size()))
		break

	###################################
	## prepare test dataset
	test_data = torchvision.datasets.MNIST(
								root=data_path,
								transform=torchvision.transforms.ToTensor(),
								train=False,
								download=download_or_not,)

	# add 1d at second col: unsqueeze, dim=1: convert (2000, 28, 28) to (2000, 1, 28, 28)
	# shrink size: up to args.test_size
	# normalize: range 0-1, by /255
	# don't calc gradient: volatile = True, so speed up process
	test_images = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:args.test_size]/255.
	test_labels = test_data.test_labels[:args.test_size]
	print("test_labels: length: {}, type: {}".format(len(test_labels), type(test_labels))) # LongTensor

	return (train_loader, test_images, test_labels)



def build_parser():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	# start: description
	parser = argparse.ArgumentParser(description='my argparse tool')
	# create a command line function
	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	#########################################################
	subparser = subparsers.add_parser('prepareData', help='prepare training and test set')
	subparser.add_argument('-mnist_dir', required=True, help="Path where mnist stored")
	subparser.add_argument('-batch_size', type=int, default=32, help="Number of samples in each batch")
	subparser.add_argument('-test_size', type=int, default=1000, help="Number of samples to test during testing")
	subparser.set_defaults(func=prepareData)

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
# python -m pdb tutorial-contents/401_prepare_mnist.py prepareData -mnist_dir /Users/Natsume/Downloads/morvan_new_pytorch/mnist -batch_size 32 -test_size 100
