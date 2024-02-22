import torch
import torch.nn as nn

# Simultaneous Gradient Descent (SimGD)
class SimGD:
    def __init__(self, policy, lr):
      self.policy = policy
      self.lr = lr

    # Empty gradients
    def zero_grad(self):
        for param in self.policy.parameters():
            if param.grad is not None:
                param.grad.detach()
                param.grad.zero_()

    # Update parameters using loss
    def step(self, loss):
        grads = torch.autograd.grad(loss, self.policy.parameters())
        for param, grad in zip(self.policy.parameters(), grads):
            param.data -= self.lr * grad