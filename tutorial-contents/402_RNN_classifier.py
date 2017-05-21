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
torchvision
"""
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 2               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
NUM_BATCHES = 100

# https://youtu.be/8SvB6B4JmfU?list=PLXO45tsB95cJxT0mL0P3-G0rBcLSvVkKH&t=137

TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

# "RNN on mnist: train an image row by row for 28 rows; "
# Mnist digital dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# when tensors wrapped by torch.Dataset, there are some transformation done above
# train_data.train_data: 0-255, ByteTensor
# train_data[0][0]: 0-1, FloatTensor
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)

# plot one example
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
# test_data.test_data: 0-255, ByteTensor
# test_data[0][0]: 0-1, FloatTensor
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())

# test dataset is feed to net box all at once, so need to shrink its size at a whole, must use test_data.test_data, not test_data[0][0]
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.

# covert to numpy array
test_y = test_data.test_labels.numpy().squeeze()[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

		# how to initialize LSTM layer
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,  # num_data each row or each step
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size): hidden line insider
        # h_c shape (n_layers, batch, hidden_size): cell line
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # [:, -1, :] is to only get r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = Variable(x.view(-1, 28, 28))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                               # batch y

		# output shape: (n_batches, input_size)
        output = rnn(b_x)                               # rnn output

		# b_y: 1-d of 0-9 values
        loss = loss_func(output, b_y)

		# clear gradients for this training step
        optimizer.zero_grad()
		# use loss to u
        loss.backward()

        optimizer.step()                                # apply gradients

        if step % 50 == 0:
			# (samples, time_step, input_size)
			# the last time step: (2000, 10)
            test_output = rnn(test_x)

			# use this output of net, to see or predict

			# test_output converted to 1d, each value 0-9
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

			# get accuracy
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, 'Batches: %4d' % step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

        if step == NUM_BATCHES:
            break
# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
