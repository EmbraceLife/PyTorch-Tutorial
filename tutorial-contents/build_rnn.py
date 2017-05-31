import argparse
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# build network in flexible way
class RNN(nn.Module):
	""" 1. create __init__; 2. create forward()
	"""
	def __init__(self):
		super(RNN, self).__init__()

		INPUT_SIZE = 28         # rnn input size / image width

		# todo: dive inside nn.LSTM
		self.rnn = nn.LSTM(
				# if use nn.RNN(), it hardly learns
			input_size=INPUT_SIZE,  # num_data each row or each step
			hidden_size=64,         # rnn hidden unit
			num_layers=1,           # number of rnn layer
			batch_first=True,       # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
			)

		self.out = nn.Linear(64, 10)

	# feed dataset to hidden layers and apply activation functions
	def forward(self, x):
		# x shape (batch, time_step, input_size)
		# r_out shape (batch, time_step, output_size)
		# h_n shape (batch, n_layers, hidden_size): hidden line insider
		# h_c shape (batch, n_layers, hidden_size): cell line
		r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

		# [:, -1, :] is to only get r_out at the last time step
		out = self.out(r_out[:, -1, :])
		return r_out, h_n, h_c, out


######################################################

def build_net(args):
	""" Build network: 1. instantiate a net; 2. build a optimizer box and loss box; 3. build net2pp for printing; 4. return (net, optimizer, loss_func, net2pp)
	"""
	######################
	# hyper parameters:
	# use args.optimizer to select an optimizer to use
	# use args.lr to set learning rate, if needed
	learning_rate = 0.01
	optimizer_select = "adam" # or 'momentum', 'adam', 'rmsprop'
	loss_select = "crossentropy" # or 'mse'

	rnn = RNN()
	######################
	## select an optimizer
	opt_SGD         = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
	opt_Momentum    = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.8)
	opt_RMSprop     = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate, alpha=0.9)
	opt_Adam        = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
	optimizers = {'sgd':opt_SGD, 'momentum':opt_Momentum, 'rmsprop':opt_RMSprop, 'adam':opt_Adam}

	optimizer = None
	for k, v in optimizers.items():
		if optimizer_select == k:
			optimizer = v

	######################
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

	return (rnn, optimizer, loss_func)


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
# python -m pdb 401_build_rnn.py build_net
