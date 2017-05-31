"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


### prepare dataset for x and target

# reproducible for random dataset created
torch.manual_seed(1)

# x data (tensor), shape=(100, 1)
# torch.unsqueeze: add 1-d to second place
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

# create noisy data with same size as x
# y data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    # build a network box
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
	# build optimizer box
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
	# build loss box
    loss_func = torch.nn.MSELoss()

	# number of training step
    steps = 0
	# train a 100 steps
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps = t + 1

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # save entire net
    torch.save(net1, '/Users/Natsume/Downloads/temp_folders/save_load/net.pkl')
	# save only the parameters
    torch.save(net1.state_dict(), '/Users/Natsume/Downloads/temp_folders/save_load/net_params.pkl')
	# save number of training step
    torch.save(steps, '/Users/Natsume/Downloads/temp_folders/save_load/steps.pkl')



def restore_net_keep_training():
    # restore entire net1 to net2
    net2 = torch.load('/Users/Natsume/Downloads/temp_folders/save_load/net.pkl')

	# continue to train
	# keep to use the same setting for optimizer
    optimizer = torch.optim.SGD(net2.parameters(), lr=0.5)
	# use the same loss function
    loss_func = torch.nn.MSELoss()

	# load previous train_step log info
    steps = torch.load('/Users/Natsume/Downloads/temp_folders/save_load/steps.pkl')

	# train a 100 steps
    epochs=100
    for t in range(epochs):
        prediction = net2(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    steps = steps + epochs
    print("we trained {} steps".format(steps))

    torch.save(net2, '/Users/Natsume/Downloads/temp_folders/save_load/net.pkl')
    torch.save(steps, '/Users/Natsume/Downloads/temp_folders/save_load/steps.pkl')
    print("save net and steps")

##################
	# feed input to net2 box, to get output
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

restore_net_keep_training()

save()

def restore_params():
    # restore model from parameters
	# first, build a model or network box with same nodes and layers
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # load parameters into the network box
    net3.load_state_dict(torch.load('net_params.pkl'))

	# feed input to network box to get output
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()
