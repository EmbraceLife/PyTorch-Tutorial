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
Video for this tutorial: https://youtu.be/CA27ONB8SQ4?list=PLXO45tsB95cJxT0mL0P3-G0rBcLSvVkKH&t=24

Dependencies:
torch: 0.1.11
matplotlib
numpy
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

############################
# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate
train_again = True  # to use previous trained model and loss

############################
# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

############################
## Create RNN class
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE, # 1 (important)
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1) # 32 link to rnn hidden layer

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

		# every step's output is applied to self.out layer, then stored into a list, then stacked into a tensor with shape (1, 10, 1)
		# 10 is num of steps
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):
			# calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


############################
## Build graph

## load rnn or create rnn from scratch
if train_again:
	rnn = torch.load("/Users/Natsume/Downloads/temp_folders/403/rnn.pkl")
else:
	rnn = RNN()
	print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()


############################
## Training

# set initial hidden state, to carry it onto next batch of training
h_state = None

# start plotting framework
plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot
plt.show()


## training loop
for step in range(60):

	## each loop, create a full steps of dataset
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

	## make dataset variables and get shape right
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

	## forward pass
    prediction, h_state = rnn(x, h_state)   # rnn output
    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # !! next step is important !!
	# repack the hidden state, break the connection from last iteration
    h_state = Variable(h_state.data)

    # plotting a step of dataset (both prediction and y_np)
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    # plt.draw() # not really necessary
    plt.pause(0.05)

plt.ioff()
plt.show()

torch.save(rnn, "/Users/Natsume/Downloads/temp_folders/403/rnn.pkl")
