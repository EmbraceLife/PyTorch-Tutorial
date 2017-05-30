import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])

# check var.grad
variable_false = Variable(tensor) # can't compute gradients
variable_true = Variable(tensor, requires_grad=True)

# add operations
t_out = torch.mean(tensor*tensor)

v_out_false = torch.mean(variable_false*variable_false)
v_out_true = torch.mean(variable_true*variable_true)

# backpropagation
# RuntimeError: there are no graph nodes that require computing gradients, due to `requires_grad=False`
# v_out_false.backward()
v_out_true.backward()
print(variable_true.grad)
print(v_out_true.creator)

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2
print(y.creator)

z = y * y * 3
out = z.mean()
out.backward() # equivalent to out.backward(torch.Tensor([1.0]))
print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2


# if grad is scalar, we can use y.backward(), if not, just provide a tensor with same size as gradients
# with grad value will be multiplied with gradients tensor provided here
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)
print(gradients)
