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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np


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
        layer1 = F.relu(self.hidden(x))
		# then feed the output of hidden relu activation to second hidden layer box, to get linear output
        layer2 = self.predict(layer1)
        return layer1, layer2

# Create an Net object with specific nodes numbers for each layer (input layer, hidden layer 1, hidden layer 2)
net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)  # net architecture

# look inside net.parameters(), it has 4 items, first two: 10 w, 10 b; second two: 10 w, 1 b
# create an optimizer SGD object based linked to all parameters
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# create a loss box
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# turn interactive mode on
plt.ion()
# make subplot title fontsize to be 'small', default is 'large'
plt.rcParams['axes.titlesize']='small'
# create a loss container
losses = []
steps = []

for t in range(100):
	# feed input x to net object to get all layer outputs
    layer1, prediction = net(x)
    # check the size of layer1 and prediction layer
	# then sent them to the list of weights and biases

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

		# access w and b of each layer
        param_names = []
        param_values = []
        for k, v in net.state_dict().items():
            param_names.append(k)
            param_values.append(v) # save as numpy array

		# now insert layer1 and prediction layer into the two list above: must use list.insert(index, value)
        param_names.insert(2, "h-layer1")
        param_names.append("pred_layer")
        param_values.insert(2, layer1.data)
        param_values.append(prediction.data)

		# add loss and step data into param_names and param_values later plotting
        losses.append(loss.data[0])
        steps.append(t)
        param_names.append("loss")
        param_values.append([steps, losses])

        num_wh_row_col = math.ceil(math.sqrt(len(param_names)))

        fig = plt.figure(1, figsize=(5, 5))
        fig.suptitle("epoch:"+str(t), fontsize="x-large")
        # plt.cla() # comment out to avoid axis printing of the last subplot
        outer = gridspec.GridSpec(num_wh_row_col, num_wh_row_col)


        param_index = 0
		# draw w and b with color
        for param in param_values:
			# add loss plot
            if param_names[param_index] == 'loss':
                num_img_row_col == 1
                s2 = 1


            elif len(param.size())==2:
				# get num_rows, num_cols of weights or bias
                s1, s2 = param.size()
                if s1 < s2:
                    num_img = s1
                    s1 = s2
                    s2 = num_img
				# num_cols are number images to have
                num_img_row_col = math.ceil(math.sqrt(s2))

            elif len(param.size()) == 1:
                num_img_row_col == 1
				# s1 = param.size() is not a float or int
                s1 = len(param)
                s2 = 1

            else:
                pass

			# make sure loss plot has no following variables
            if param_names[param_index] != 'loss':
			    # consider a col is an image, num_rows is all pixels of an image
	            img_wh = math.ceil(math.sqrt(s1))


				# add pixels to fill a square
	            missing_pix = img_wh*img_wh - s1
				# for each col, add enough 0s to met a square of pixels
	            param_padded = torch.cat((param.view(s1,s2), torch.zeros((missing_pix, s2))),0)


            # plt.cla()
            # fig, sub = plt.subplots(num_img_row_col, num_img_row_col)
            inner = gridspec.GridSpecFromSubplotSpec(num_img_row_col, num_img_row_col, subplot_spec=outer[param_index], wspace=0.0, hspace=0.0)



            for index in range(s2):

                ax = plt.Subplot(fig, inner[index])

                if param_names[param_index] == 'loss':

    				# plot loss
                    ax.plot(param[0], param[1], 'b-')
					# text location coordinates changes as axes limits changes
					# coordinates are to be consistent with the subplot x and y axes
                    # ax.text(20, 0.3, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
					# if we contrain xlim and ylim, then text coordinates won't change as axes don't change any more
                    ax.set_xlim((0,100))
                    ax.set_ylim((0,0.35))
                    ax.set_title("loss: %.4f" % loss.data[0], fontdict={'size': 5, 'color':  'red'})
                    fig.add_subplot(ax)

                else:
                    ax.imshow(np.reshape(param_padded.numpy()[:, index], (img_wh, img_wh)), cmap='gray')
				# How to change subplot title's size??? check doc and examples online
				# how to handle subplot main title???
                    if s2 > 1:
					# make sure the title is in the middle
                        if index == int(num_img_row_col/2):
                            ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape))
                    else:
                        ax.set_title(param_names[param_index]+": {}".format(param.numpy().shape))
                    ax.set_xticks(())
                    ax.set_yticks(())
                    fig.add_subplot(ax)

            param_index += 1

            # Pause for *interval* seconds
        plt.pause(0.5)
        plt.cla()		



        # # clear the current axes
		# # in fact it is to clear the plots for the last plotted plot
        # plt.cla()
		# # clear the first plotted plot
        # plt.subplot(121).cla()
		#
		# # store loss values
        # plt.subplot(121)
        # losses.append(loss.data[0])
        # steps.append(t)
		# # plot loss
        # plt.plot(steps, losses, 'b-')
		# # text location coordinates changes as axes limits changes
		# # coordinates are to be consistent with the subplot x and y axes
        # plt.text(20, 0.3, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
		# # if we contrain xlim and ylim, then text coordinates won't change as axes don't change any more
        # plt.xlim((0,100))
        # plt.ylim((0,0.35))
        # plt.title("loss in training")
		#
        # # plt.subplot(121).cla()
        # # plt.subplot(122).cla()
		# # create a subplot with position inside the figure
        # plt.subplot(122)
		# # plot features and targets
        # plt.scatter(x.data.numpy(), y.data.numpy())
		# # plot features and predictions
        # plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.title("(features,target) vs (features, prediction)")
		# # plot texts


		# # Pause for *interval* seconds
        # plt.pause(0.5)




plt.ioff()
plt.show()

# "train a regression model, plot while training, torch.manual_seed, torch.unsqueeze, torch.linspace, torch.pow, torch.pow, torch.rand, Net(torch.nn.Module), torch.nn.Linear, F.reul, print(net), net.parameters(), torch.optim.SGD, torch.nn.MSEloss, optimizer.zero_grad, loss.backward, optimizer.step, plt.ion, plt.ioff, plt.cla, plt.pause"
