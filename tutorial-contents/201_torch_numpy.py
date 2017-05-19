"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
numpy
"""
import torch
import numpy as np

# details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations

# create a simple 2-d numpy array
np_data = np.arange(6).reshape((2, 3))

# convert numpy array to pytorch tensor
torch_data = torch.from_numpy(np_data)

# convert a pytorch tensor to numpy array
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)


# create a pytorch tensor from a list
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
	# numpy and pytorch operations: abs
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

# sin
print(
    '\nsin',
	# # numpy and pytorch operations: sin
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean
print(
    '\nmean',
	# numpy and pytorch operations: mean
	# torch.mean(a, 0): to shrink rows
	# torch.mean(a, 1): to shrink cols
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

# matrix multiplication
# create 2-d pytorch tensor from list
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
	# numpy, pytorch matrix multiplication
	# (1, 4), (4, 5) = (1, 5)
	# dot(row_tensor1, col_tensor2) = multiply and sum
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)
# incorrect method
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
	# dot operation means different things for pytorch and numpy
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]]
    '\ntorch: ', tensor.dot(tensor)     # tensor will be flatten to 1-d, so you'll get 30.0
)
