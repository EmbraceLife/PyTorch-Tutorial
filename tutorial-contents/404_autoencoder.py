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
numpy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


torch.manual_seed(1)    # reproducible

###################################
# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

###################################
## Load dataset
# do check doc for this func
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

## plot one example
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[2])
plt.show()

## make batch
# explore object: train_loader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#############################
## build autoEncoder class:
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

		# create encoder: a sequential object
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
			# encoder or shrink down to 3 features
        )

		# create decoder: a sequential object
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

	# apply dataset onto this neuralnet
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

##########################
## Build graph framework
# instantiate autoEncoder
autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
# mse: not compare labels (crossentropy), but compare image values
loss_func = nn.MSELoss()


############################
## create images comparison framework
# subplotting (2 rows, 5 cols)
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot
plt.show()

# prepare 5 images for plotting and make them a Variable
view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.)

# plot 5 images on a same row
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

#########################
## training
# for each epoch
for epoch in range(EPOCH):

	# for each batch, batch features and batch label
    for step, (x, y) in enumerate(train_loader):
		# first, check x, y, size
		# x.size(): (64, 1, 28, 28)

		# make batch x a Variable, shape (batch, 28*28)
        b_x = Variable(x.view(-1, 28*28))
		# make batch y a Variable, shape (batch, 28*28)
        b_y = Variable(x.view(-1, 28*28))
		# batch label
        b_label = Variable(y)


		## actual forward pass
        encoded, decoded = autoencoder(b_x)
		## loss on mse, using no labels
        loss = loss_func(decoded, b_y)

		## actual backward pass
        optimizer.zero_grad() # clear gradients
        loss.backward() # update gradients
        optimizer.step() # update weights

		## after 100 batches training, log and plot decoder images
        if step % 100 == 0:

			# print: epoch | batch | loss
            print('Epoch: ', epoch, '| Batch: ', step, '| train loss: %.4f' % loss.data[0])

            # plotting decoded image on second row
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()


##########################
## visualize in 3D plot
# not sure the need for 3D plotting yet????


## prepare 200 images
view_data = Variable(train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.)

## encode these 200 images
# encoder shape (200, 3)
encoded_data, _ = autoencoder(view_data)

# create a second plot
fig = plt.figure(2)
ax = Axes3D(fig)
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
	# cm is a module from matplotlib
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
