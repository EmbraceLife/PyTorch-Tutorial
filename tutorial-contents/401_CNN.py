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
torchvision
matplotlib
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
# to load data
import torch.utils.data as Data
# to handle images
import torchvision
import matplotlib.pyplot as plt

# reproducible
torch.manual_seed(1)

# Hyper Parameters
# train the training data n times, to save time, we just train 1 epoch
EPOCH = 2
BATCH_SIZE = 50
NUM_BATCHES = 100
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False   # set to False if you have downloaded already


# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
	# where to save the data or load the data from
    root='./mnist/',
	# this is training data
    train=True,
	# Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
	# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    transform=torchvision.transforms.ToTensor(),

	# download it if you don't have it already
    download=DOWNLOAD_MNIST,
)
# it is possible to hack torchvision.datasets.MNIST and torchvision.transforms.ToTensor()


# dr train_data to see all its attr and methods
# (60000, 28, 28)
print(train_data.train_data.size())
print(train_data.train_labels.size())               # (60000)

# plot a single image with gray color scale
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# train_data must be TensorDataset first already, although its class is torchvision.datasets.mnist.MNIST
# in practice, train_data[0] return a tuple of a single pair of dataset (image, lable)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# once TensorDataset turned into DataLoader, we can only access it by enumerate()

# convert test data into Variable, pick 2000 samples to speed up testing
# test_data.test_data type is ByteTensor, so is train_data.train_data
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# hand_tranform data: shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

		# how to build cnn layer? use fast method with layer class
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

# the params for defining CNN is embedded inside, see above
cnn = CNN()
print(cnn)  # net architecture

# build optimizer box for all params or weights
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# build loss box
loss_func = nn.CrossEntropyLoss()
 # the target label is not one-hotted ????

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
		# gives batch data,
		# only when iterate train_loader, train_data.train_data is normalized from 0-255 to 0-1
		# train_data.train_labels still 0-9
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

		# feed variable input to cnn object, to get output of forward()
        output = cnn(b_x)
		# feed output and true label to get loss
        loss = loss_func(output, b_y)
		# clear gradients for this training step
        optimizer.zero_grad()
        # use loss value to calc and update gradients
        loss.backward()
		# use new gradients to update parameters for optimizer and net
        optimizer.step()

		# every 50 batches training (including at the first batch), do a test
        if step % 50 == 0:
			# feed cnn box with test data, get output
            test_output = cnn(test_x)
			# output'shape (n, 10)
			# take max value of each row by columns
			# get not the max value, but the index of the max value in the row
			# make sure the pred_y is 1-d tensor
            pred_y = torch.max(test_output, 1)[1].data.squeeze()

			# calc accuracy
            accuracy = sum(pred_y == test_y) / test_y.size(0)

			# print epoch, num_batches, loss and accuracy out nicely
            print('Epoch: ', epoch, '| num_batches: %4d' % step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

		# to speed up the process, set number of batches to train, not the full train_data
        if step == NUM_BATCHES:
            break


# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
