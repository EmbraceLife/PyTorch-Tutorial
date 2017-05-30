import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])
variable_false = Variable(tensor) # can't compute gradients
variable_true = Variable(tensor, requires_grad=True)

# tensor operations
t_out = torch.mean(tensor*tensor)
# variable operations
v_out_false = torch.mean(variable_false*variable_false)
v_out_true = torch.mean(variable_true*variable_true)

"""
# backpropagation
v_out_false.backward()
# RuntimeError: there are no graph nodes that require computing gradients, due to `requires_grad=False`
"""
# to do backward for a second or third time, must keep `retain_variables=True` every time when call backward
v_out_true.backward(retain_variables=True)
# equivalent to v_out_true.backward(torch.ones(1))
print(variable_true.grad)
print(v_out_true.creator) # future: v_out_true.grad_fn
v_out_true.backward(retain_variables=True)
# every time call backward() will change the gradients
print(variable_true.grad)
print(v_out_true.creator) # future: v_out_true.grad_fn
# use torch.autograd's backward
v_out_true.backward(torch.ones(1), retain_variables=True) # future: retain_graph=True, create_graph=True
print(variable_true.grad)

"""
Variable containing:
 0.5000  1.0000
 1.5000  2.0000
[torch.FloatTensor of size 2x2]

<torch.autograd._functions.reduce.Mean object at 0x116a159e8>
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]

<torch.autograd._functions.reduce.Mean object at 0x116a159e8>
Variable containing:
 1.5000  3.0000
 4.5000  6.0000
[torch.FloatTensor of size 2x2]
"""

x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2
# we can set gradients freely I think?
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
while y.data.norm() < 1000:
    y = y * 2
    y.backward(gradients, retain_variables=True)
    print(x.grad)
