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

View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
from torch.autograd import Variable
# for quickly access all activation functions to run with data
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# create tensor (100, 2) of 1s
n_data = torch.ones(100, 2)

# create normal distribution of tensor size as n_data
# each value is based on 2*n_data and vary by std 1
# see examples
# torch.normal(means=torch.ones(10), std=0.1)
# torch.normal(means=torch.arange(1, 11), std=1)
# now the x0 is feature, size (100, 2), normal values (mean=2, std=1)
x0 = torch.normal(2*n_data, 1)
# y0 is target or label, all set to 0s, size(100)
y0 = torch.zeros(100)

# x1 is also feature, size (100, 2), normal values (mean=-2, std=1)
x1 = torch.normal(-2*n_data, 1)
# y1 is target or label, all set to 1s, size(100)
y1 = torch.ones(100)

# torch.cat: x1 and x0 must be same type and same size, in a tuple
# 0: to add by rows ; 1: to add by cols; default is 0, by rows
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)
# shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# x is 2-d dataset, first col is set as x, second col set as y to plot, using labels to color the points
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

# define the network box: input: (n, 2), hidden: (2, 10) ?????
net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)  # net architecture

# build optimizer box: use SGD; lr set small to train slowly
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

# build loss box: use CrossEntropyLoss
loss_func = torch.nn.CrossEntropyLoss()
# the target label is NOT an one-hotted

# plt.ion()   # something about plotting
# plt.show()

for t in range(100):
	# feed dataset feature to network box, return prediction or output
    out = net(x)
	# feed prediction and true label to loss box, return loss value
	# Note: prediction comes first before true label
    loss = loss_func(out, y)
	# Note: the prediction or output is NOT one-hotted

	# before update parameters' gradients, clear out gradients
    optimizer.zero_grad()
	# by now gradients is None
	# use latest loss value to update gradients
    loss.backward()
	# now we have new gradients
	# use updated gradients to update weights or parameters, and store in both optimizer and net objects
    optimizer.step()
	# now parameters are updated

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
		# torch.max return 2 tensors, first: tensor of maximum values; second: tensor of index of maximum values
        prediction = torch.max(F.softmax(out), 1)[1]
		# make sure it is 1-d, squeezed from 2-d
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
		# calc accuracy
        accuracy = sum(pred_y == target_y)/200
		# print a text on accruacy
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
