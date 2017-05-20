"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# input args: int or long as you like
torch.manual_seed(1)    # reproducible

# pytorch want all 1-d input become 2-d input
# >>> x = torch.Tensor([1,2,3,4])
# >>> x.size()
# torch.Size([4])
# >>> x.view(1,4).size()
# torch.Size([1, 4])
# >>> x.view(4,1).size()
# torch.Size([4, 1])
# >>> torch.unsqueeze(x, 0).size()
# torch.Size([1, 4])
# >>> torch.unsqueeze(x, 1).size()
# torch.Size([4, 1])
# >>> torch.unsqueeze(x, -1).size()
# torch.Size([4, 1])
# >>> torch.unsqueeze(x, -2).size()
# torch.Size([1, 4])
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

# torch.pow(x, 2), torch.rand(a set of int)
# create a random noise
y = x.pow(2) + 0.2*torch.rand(x.size())

# torch can only train on Variable
x, y = Variable(x), Variable(y)

# simple dataset like this can be plotted together
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# to build a net class, must inherit from Module class (all network modules)
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
		# first hidden layer
		# torch.nn.Linear(input_features, output_features)
		# create a linear layer class (still empty box)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
		# second hidden layer (empty box without data)
        self.predict = torch.nn.Linear(n_hidden, n_output)
		# return output layer

    def forward(self, x):
		# feed data into first hidden layer box
		# also apply relu activation directly onto the feeded hidden layer
        x = F.relu(self.hidden(x))
		# then feed the output of hidden relu activation to second hidden layer box, to get linear output
        x = self.predict(x)
        return x

# Create an Net object with specific nodes numbers for each layer (input layer, hidden layer 1, hidden layer 2)
net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)  # net architecture

# look inside net.parameters(), it has 4 items, first two: 10 w, 10 b; second two: 10 w, 1 b
# create an optimizer SGD object based linked to all parameters
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# create a loss box
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# turn interactive mode on
plt.ion()   # something about plotting
plt.show()

for t in range(100):
	# feed input x to net object to get final output prediction of net.forward()
    prediction = net(x)

	# feed forward output and true target values to loss box
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

	# clear gradients for next train
    optimizer.zero_grad()

	# backpropagation, compute gradients
    loss.backward()

# 	Question: what exactly does `optimizer.step()` do? (following is my understanding)
# 1. its doc says: Performs a single optimization step (not explaining anything to me)
# 2. its source says: step() contains A closure that reevaluates the model and returns the loss. (fine, but ...)
# 3. when and why do we need to use a closure here to reevaluate model and get loss?
# 4. besides the closure, `step()` is to update parameters or weights and store them inside optimizer, right? Also the updated weights or parameters must also be stored inside `net`, otherwise we won't get new prediction and new loss every loop. right?
    optimizer.step()

    if t % 5 == 0:
        # clear the current axes
        plt.cla()
		# plot features and targets
        plt.scatter(x.data.numpy(), y.data.numpy())
		# plot features and predictions
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		# plot texts
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
		# Pause for *interval* seconds
        plt.pause(0.5)

plt.ioff()
plt.show()

# "train a regression model, plot while training, torch.manual_seed, torch.unsqueeze, torch.linspace, torch.pow, torch.pow, torch.rand, Net(torch.nn.Module), torch.nn.Linear, F.reul, print(net), net.parameters(), torch.optim.SGD, torch.nn.MSEloss, optimizer.zero_grad, loss.backward, optimizer.step, plt.ion, plt.ioff, plt.cla, plt.pause"
