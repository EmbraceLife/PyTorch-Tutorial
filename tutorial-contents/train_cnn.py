# python -m pdb tutorial-contents/train_cnn.py train -mnist_dir /Users/Natsume/Downloads/morvan_new_pytorch/mnist -batch_size 32 -test_size 100 -num_epochs 10 -num_batches 50 -net /Users/Natsume/Downloads/temp_folders/train_cnn/net.pkl -log /Users/Natsume/Downloads/temp_folders/train_cnn/log.pkl

import argparse
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

from prepare_mnist import prepareData
from build_cnn import build_net

def train(args):
	""" Trains a model.
	"""
	## hyper-parameters
	train_again = True
	plot_loss = True

	# prepare dataset
	train_loader, test_images, test_labels = prepareData(args)

	# total number of batches of an epoch
	total_train_samples = train_loader.dataset.train_data.__len__()
	total_num_batches = int(total_train_samples / args.batch_size)

	## load cnn or create cnn from scratch
	if train_again:
		cnn = torch.load(args.net_path)
		optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
		loss_func = nn.CrossEntropyLoss()
		steps, losses_train, losses_val, accuracies = torch.load(args.log_path)
		previous_step = steps[-1]
	else:
		cnn, optimizer, loss_func = build_net(args)
		# store every 10 number of batches trainning
		losses_val = []
		losses_train = []
		accuracies = []
		steps = []
		previous_step = 0

	# for every epoch of training
	for epoch_idx in range(args.num_epochs):
		# store every batch loss
		loss_list = []

		# loop every batch of an epoch
		for batch_idx, (batch_img, batch_lab) in enumerate(train_loader):
			# rnn model only access input with 3d shape (batch, 28, 28)
			# cnn needs size (batch, 1, 28, 28)
			# but both rnn and cnn need to work with variables not tensors
			b_img = Variable(batch_img)
			b_lab = Variable(batch_lab)
			# actual batch training
			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, out = cnn(b_img)
			loss = loss_func(out, b_lab)
			optimizer.zero_grad()
			loss.backward()
			## todo: b_img is required_grad=True right?
			optimizer.step()
			# store every batch loss
			loss_list.append(loss.data.numpy()[0])

			# every 10 batches, print log; use args.num_batches_log
			if batch_idx % 10 == 0:
				# store steps index for every 10 batches
				steps.append(args.num_batches * epoch_idx + batch_idx + previous_step)

				# store avg_loss every 10 batches training
				avg_loss_batch_train = np.array(loss_list).mean()
				loss_list = []
				losses_train.append(avg_loss_batch_train)

				# cnn takes (batch, 1, img_width, img_height), no need to shrink tensor.size here
				conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, test_output = cnn(test_images)
				# validation loss

				loss_val = loss_func(test_output, test_labels)
				losses_val.append(loss_val.data.numpy()[0])
				# use test|validation set, get accuracy
				pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
				accuracy = sum(pred_y == test_labels.data.numpy()) / test_labels.data.numpy().shape[0]
				accuracies.append(accuracy)

				# print log
				print('Epoch: '+ str(epoch_idx+1) + ' Batches: %04d' % (batch_idx+1) + ' | avg_train loss: %.4f' % avg_loss_batch_train, 'loss_val: %.4f'%loss_val.data.numpy()[0], ' | test accuracy: %.2f' % accuracy)

			# If don't want to train for a full epoch, stop early on specific number of batches
			if args.num_batches-1 == batch_idx:
				break

	# save net and log
	torch.save(cnn, args.net_path)
	torch.save((steps, losses_train, losses_val, accuracies), args.log_path)

	if plot_loss:
		plt.plot(steps, losses_train, c='blue', label='train')
		plt.plot(steps, losses_val, c='red', label='val')
		plt.legend(loc='best')
		plt.show()

#########################################################
def build_parser():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	# start: description
	parser = argparse.ArgumentParser(description='my argparse tool')
	# create a command line function
	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	#########################################################
	subparser = subparsers.add_parser('train', help='Trains a model for the first time.')
	# add args to train function
	subparser.add_argument('-mnist_dir', required=True, help="Path where mnist stored")
	subparser.add_argument('-batch_size', type=int, default=32, help="Number of samples in each batch")
	subparser.add_argument('-num_batches', type=int, default=100, help="Number of batches to train in each epoch")
	subparser.add_argument('-test_size', type=int, default=1000, help="Number of samples to test during testing")
	subparser.add_argument('-net', '--net_path', required=True, help="Path to save neuralnet model")
	subparser.add_argument('-log', '--log_path', required=True, help="Path to save log information: losses, steps")
	subparser.add_argument('-num_epochs', type=int, default=1, help="Number of epochs to train this time")

	subparser.set_defaults(func=train)

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
# python -m pdb 401_build_rnn.py build_net
