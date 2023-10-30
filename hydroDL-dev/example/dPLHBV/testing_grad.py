import torch
import torch.autograd as autograd

# Define the function y(t) = sin(t)
def y_function(t):
    return torch.sin(t)

# Choose the value of t at which you want to compute the gradient
t_value = torch.tensor([0.5], requires_grad=True)  # Adjust the value as needed

# Compute the output y(t)
y_t = y_function(t_value)

# Compute the gradient using autograd.grad()
gradient = autograd.grad(outputs=y_t, inputs=t_value, grad_outputs=torch.ones_like(y_t), retain_graph=True)
print(gradient)
# gradient contains the gradient of y(t) with respect to t
