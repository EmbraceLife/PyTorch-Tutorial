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
## to display plotting
python -m pdb tutorial-contents/103_argparse_train_again.py train -net /Users/Natsume/Downloads/temp_folders/103/net.pkl -log /Users/Natsume/Downloads/temp_folders/103/log.pkl -p /Users/Natsume/Downloads/temp_folders/103 -d

## to save plots (without pdb)
python tutorial-contents/103_argparse_train_again.py train -net /Users/Natsume/Downloads/temp_folders/103/net.pkl -log /Users/Natsume/Downloads/temp_folders/103/log.pkl -p /Users/Natsume/Downloads/temp_folders/103 -num 200

## continue to train (save plots), without pdb
python tutorial-contents/103_argparse_train_again.py train_again -net /Users/Natsume/Downloads/temp_folders/103/net.pkl -log /Users/Natsume/Downloads/temp_folders/103/log.pkl -p /Users/Natsume/Downloads/temp_folders/103 -num 200
"""

###########################
# style from ONMT combined with kur style
###########################

import argparse
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
import os
import subprocess

# this framework (py.file) can run individual functions like train() or train_again() independently

def prepareData(args):
	""" Prepare dataset for training later: 1. return x, y as Variables; 2. args: can help do lots of things: shrink data, special treatment to data, ...
	"""
	# reproducible
	torch.manual_seed(1)
	# make a 1-d tensor 2-d
	x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
	# make random y based on x
	y = x.pow(2) + 0.2*torch.rand(x.size())
	# convert tensor to variables
	x, y = Variable(x), Variable(y)

	return (x, y)

# move Net in global env due to AttributeError: Can't pickle local object 'build_net.<locals>.Net'
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):

        layer1 = F.relu(self.hidden(x))
        layer2 = self.predict(layer1)

        return layer1, layer2

def build_net(args):
	""" Build network: 1. Create Net Class with its forward pass; 2. instantiate a net; 3. build a optimizer box and loss box; 4. args: to bring in parameters for build nets
	"""

	# Net(n_feature=args.input_size, n_hidden=args.hidden_size, n_output=args.out_size)
	net = Net(n_feature=1, n_hidden=10, n_output=1)
	print(net)  # net architecture

	# lr = args.lr
	optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

	loss_func = torch.nn.MSELoss()
	return (net, optimizer, loss_func)

def saveplots(args, param_names, param_values):

	# fig: num_row_img, num_col_img
	num_wh_row_col = math.ceil(math.sqrt(len(param_names)))

	fig = plt.figure(1, figsize=(6, 6))

	epoch = param_values[-1][0][-1]
	fig.suptitle("x, y, 1-d regression, epoch:"+str(epoch), fontsize="x-large")
	# create subplots on fig
	outer = gridspec.GridSpec(num_wh_row_col, num_wh_row_col)


	param_index = 0

	# set up s1, s2, num_img_row_col for diff param
	for param in param_values:

		# loss plotting is diff from all other param plotting
	    if param_names[param_index] == 'loss':

			# subplot: num_row_img, num_col_img
	        num_img_row_col == 1
	        s2 = 1

		# for all other param, if dim == 2
	    elif len(param.size())==2:

	        s1, s2 = param.size()

			# if s2 is large, swap values between s1 and s2, make s1 larger
	        if s1 < s2:
	            num_img = s1
	            s1 = s2
	            s2 = num_img

	        num_img_row_col = math.ceil(math.sqrt(s2))

		# for all other param, if dim == 1
	    elif len(param.size()) == 1:
			# set subplot num_row_img == 1
	        num_img_row_col == 1

	        s1 = len(param)
	        s2 = 1

	    else:
	        pass

		# all param other than loss must have img_wh, param_padded
	    if param_names[param_index] != 'loss':
	        img_wh = math.ceil(math.sqrt(s1))

	        missing_pix = img_wh*img_wh - s1
	        param_padded = torch.cat((param.view(s1,s2), torch.zeros((missing_pix, s2))),0)

		# create sub-subplots grid on a subplot
	    inner = gridspec.GridSpecFromSubplotSpec(num_img_row_col, num_img_row_col, subplot_spec=outer[param_index], wspace=0.0, hspace=0.0)

		# loop every sub-subplots of a subplot or param
	    for index in range(s2):

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

			# plot other param other than loss
	        else:
				# plot a single sub-subplot of subplot
	            ax.imshow(np.reshape(param_padded.numpy()[:, index], (img_wh, img_wh)), cmap='gray')

				# if there more than 1 sub-subplots in a subplot
				# where to draw title
	            if s2 > 1:
	                if index == int(num_img_row_col/2):
	                    ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				# if just 1 sub-subplot in a subplot, use default title position
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
def display(args, param_names, param_values):

	# fig: num_row_img, num_col_img
	num_wh_row_col = math.ceil(math.sqrt(len(param_names)))

	fig = plt.figure(1, figsize=(6, 6))

	epoch = param_values[-1][0][-1]
	fig.suptitle("x, y, 1-d regression, epoch:"+str(epoch), fontsize="x-large")
	# create subplots on fig
	outer = gridspec.GridSpec(num_wh_row_col, num_wh_row_col)


	param_index = 0

	# set up s1, s2, num_img_row_col for diff param
	for param in param_values:

		# loss plotting is diff from all other param plotting
	    if param_names[param_index] == 'loss':

			# subplot: num_row_img, num_col_img
	        num_img_row_col == 1
	        s2 = 1

		# for all other param, if dim == 2
	    elif len(param.size())==2:

	        s1, s2 = param.size()

			# if s2 is large, swap values between s1 and s2, make s1 larger
	        if s1 < s2:
	            num_img = s1
	            s1 = s2
	            s2 = num_img

	        num_img_row_col = math.ceil(math.sqrt(s2))

		# for all other param, if dim == 1
	    elif len(param.size()) == 1:
			# set subplot num_row_img == 1
	        num_img_row_col == 1

	        s1 = len(param)
	        s2 = 1

	    else:
	        pass

		# all param other than loss must have img_wh, param_padded
	    if param_names[param_index] != 'loss':
	        img_wh = math.ceil(math.sqrt(s1))

	        missing_pix = img_wh*img_wh - s1
	        param_padded = torch.cat((param.view(s1,s2), torch.zeros((missing_pix, s2))),0)

		# create sub-subplots grid on a subplot
	    inner = gridspec.GridSpecFromSubplotSpec(num_img_row_col, num_img_row_col, subplot_spec=outer[param_index], wspace=0.0, hspace=0.0)

		# loop every sub-subplots of a subplot or param
	    for index in range(s2):

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

			# plot other param other than loss
	        else:
				# plot a single sub-subplot of subplot
	            ax.imshow(np.reshape(param_padded.numpy()[:, index], (img_wh, img_wh)), cmap='gray')

				# if there more than 1 sub-subplots in a subplot
				# where to draw title
	            if s2 > 1:
	                if index == int(num_img_row_col/2):
	                    ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape), fontdict={'size': 8, 'color':  'black'})
				# if just 1 sub-subplot in a subplot, use default title position
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
	x, y = prepareData(args)

	net=None
	# build net
	net, optimizer, loss_func = build_net(args)

	# train
	losses = []
	steps = []

	if args.display:
		plt.ion()

	for t in range(args.num_epochs):

		layer1, prediction = net(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# plots and save every 5 steps
		if t % 5 == 0:
			losses.append(loss.data.numpy().tolist()[0])
			steps.append(t)

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
			param_values.insert(0, y.data)
			param_values.insert(0, x.data)

			# losses.append(loss.data[0])
			# steps.append(t)
			param_names.append("loss")
			param_values.append([steps, losses])

			if args.display:
				display(args, param_names, param_values)

			else:
				saveplots(args, param_names, param_values)

	if args.display:
		plt.ioff()
	else:
		# save net and log
		torch.save(net, args.net_path)
		torch.save((steps, losses), args.log_path)
		# convert saved images to gif (speed up, down, normal versions)
		img2gif(args)

def train_again(args):
	""" Trains a model.
	"""
	# prepare dataset
	x, y = prepareData(args)

	# load net and log
	net = torch.load(args.net_path)
	steps, losses = torch.load(args.log_path)

	previous_steps = steps[-1]

	# build workflow
	optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
	loss_func = torch.nn.MSELoss()

	# train
	if args.display:
		plt.ion()

	for t in range(args.num_epochs):

		layer1, prediction = net(x)
		loss = loss_func(prediction, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# plots and save every 5 steps
		if t % 5 == 0:
			# add new loss onto the list losses
			losses.append(loss.data.numpy().tolist()[0])
			# add newly trained steps from previous_steps
			# do not add 1 as we start training at epoch_0.png
			steps.append(previous_steps+t)

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
			param_values.insert(0, y.data)
			param_values.insert(0, x.data)

			param_names.append("loss")
			param_values.append([steps, losses])

			if args.display:
				display(args, param_names, param_values)

			else:
				saveplots(args, param_names, param_values)

	if args.display:
		plt.ioff()
	else:
		# update net and log
		torch.save(net, args.net_path)
		torch.save((steps, losses), args.log_path)
		# convert saved images to gif (speed up, down, normal versions)
		img2gif(args)


def build_parser():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	# start: description
	parser = argparse.ArgumentParser(description='my argparse tool')

	# create a command line function
	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	# the command line function defined
	subparser = subparsers.add_parser('train', help='Trains a model for the first time.')

	# the command line function's arguments
	subparser.add_argument('-d', '--display', action='store_true',
		help='Plot whole process while training')

	subparser.add_argument('-net', '--net_path', required=True,
	                    help="Path to save neuralnet model")

	subparser.add_argument('-log', '--log_path', required=True, help="Path to save log information: losses, steps")

	subparser.add_argument('-p', '--plots_path', required=True, help="Path to save plots")

	# input Int
	subparser.add_argument('-num', '--num_epochs', type=int, default=100,
	                    help="Number of epochs to train this time")

	subparser.add_argument('-s', '--selection',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')

	subparser.set_defaults(func=train)

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
	os.chdir('/Users/Natsume/Downloads/temp_folders/103')

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
