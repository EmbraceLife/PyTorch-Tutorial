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

###########################
# style from ONMT combined with kur style
###########################

import argparse
import sys

# this framework (py.file) can run individual functions like train() or train_again() independently
def train(args):
	""" Trains a model.
	"""
	print("I am running in train(), then exit")

def train_again(args):
	""" Trains a model.
	"""
	print("I am training the model again, then exit")

def build_parser():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	# start: description
	parser = argparse.ArgumentParser(description='my argparse tool')

	# create a command line function
	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	# the command line function defined
	subparser = subparsers.add_parser('train', help='Trains a model.')

	# the command line function's arguments
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')

	subparser.add_argument('-train_src', required=True,
	                    help="Path to the training source data")

	subparser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
	# input Int
	subparser.add_argument('-seq_length', type=int, default=50,
	                    help="Maximum sequence length")

	subparser.add_argument('-t', '--target',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')

	subparser.set_defaults(func=train)

	# the command line function defined as train_again
	subparser = subparsers.add_parser('train_again', help='Trains a model.')
	subparser.set_defaults(func=train_again)

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
