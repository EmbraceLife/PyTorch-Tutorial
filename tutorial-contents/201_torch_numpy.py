import torch
import numpy as np

# from np.array to torch.tensor, and backforth
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

# from list to FloatTensor
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point

# operations on numpy and tensor
np.abs(data)
torch.abs(tensor)

np.sin(data)
torch.sin(tensor)

# numpy and pytorch operations: mean
# torch.mean(a, 0): to shrink rows
# torch.mean(a, 1): to shrink cols
np.mean(data)
torch.mean(tensor)

# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# numpy, pytorch matrix multiplication
# (1, 4), (4, 5) = (1, 5)
# dot(row_tensor1, col_tensor2) = multiply and sum
np.matmul(data, data),     # [[7, 10], [15, 22]]
torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]

# incorrect method
data = np.array(data)
data.dot(data)        # [[7, 10], [15, 22]]
tensor.dot(tensor)     # tensor will be flatten to 1-d, so you'll get 30.0

###################### based on Pytorch official tutorials
# Construct a 5x3 matrix, uninitialized:
x = torch.Tensor(5, 3)
# Construct a randomly initialized matrix
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# Addition: giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)

y.add_(x)  # Addition: in-place
x.copy_(y)
x.t_()     # all these will change ``x``.

# You can use standard numpy-like indexing with all bells and whistles!
print(x[:, 1])

## when tensor a changes, numpy b changes too
a = torch.ones(5)
b = a.numpy()
a.add_(1)

## when numpy array a changes, tensor b changes too
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)

## use cuda
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
