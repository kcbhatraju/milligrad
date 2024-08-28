import numpy as np

from .utils import epsilon


class SGD:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    
    def step(self):
        for param in self.params:
            param.update_item(-self.lr * param.grad())

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999):
        self.params = params
        self.initial_lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.grad_avg = [0. for _ in params]
        self.grad_sq_avg = [0. for _ in params]

        self.t = 0.
    
    def step(self):
        self.t += 1.
        self.curr_grads = [param.grad() for param in self.params]
        
        for i, (param, grad) in enumerate(zip(self.params, self.curr_grads)):
            self.grad_avg[i] = self.beta1 * self.grad_avg[i] + (1 - self.beta1) * grad
            self.grad_sq_avg[i] = self.beta2 * self.grad_sq_avg[i] + (1 - self.beta2) * grad ** 2

            curr_lr = np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t) * self.initial_lr
            param.update_item(-curr_lr * self.grad_avg[i] / (np.sqrt(self.grad_sq_avg[i]) + epsilon))
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
