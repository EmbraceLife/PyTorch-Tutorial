"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
# with torch.nn.functional as F, we can quickly deploy activations to use
# to see detailed doc of activation, we need
# import torch.nn as nn
# doc nn.ReLU
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# torch.linspace(range_start, range_end, total_num)
# it is 1-d tensor
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x)
x_np = x.data.numpy()   # numpy array for plotting

# apply activation only to a Variable, not tensor
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
# F.softmax(x),  x has to be 2-d tensor
# y_softmax = F.softmax(x)  softmax is a special kind of activation function, it is about probability


# create a single figure with a particular size
plt.figure(1, figsize=(8, 6))
# create a subplot with position inside the figure
plt.subplot(221)
# plot line with color and label
plt.plot(x_np, y_relu, c='red', label='relu')
# set y limit
plt.ylim((-1, 5))
# set legend with location
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
