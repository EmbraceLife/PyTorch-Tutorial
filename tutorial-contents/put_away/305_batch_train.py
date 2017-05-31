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
"""
import torch
# import utils.data module to help
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

# BATCH_SIZE = 5
BATCH_SIZE = 8

# create tensor of 10 ascending and descending
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

# wrap two data tensors together and using index to retrieve them
# how to access dataset
# torch_dataset[0], torch_dataset[1] ...
# [ td for td in torch_dataset]
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# put whole dataset into batches
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
    drop_last=False				# True: drop smaller remaining data; False: keep smaller remaining data
)

 # train entire dataset 3 times
for epoch in range(3):
	# how loader is accessed for each batch of data
    for step, (batch_x, batch_y) in enumerate(loader):
		# for each training step
        # train your data...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
