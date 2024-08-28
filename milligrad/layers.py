import numpy as np

from .alg import matmul, relu, sigmoid, softmax, tanh
from .backward import Tensor_, to_numpy
from .initializers import _Initializer
from .utils import epsilon


class Activation:
    def __init__(self, func="linear"):
        self.func = func
        self.params = []
    
    def __call__(self, x, training=True):
        if self.func == "linear":
            self.output = x
        elif self.func == "relu":
            self.output = relu(x)
        elif self.func == "tanh":
            self.output = tanh(x)
        elif self.func == "sigmoid":
            self.output = sigmoid(x)
        elif self.func == "softmax":
            self.output = softmax(x)

        return self.output

    def parameters(self):
        return self.params

class Dense:
    def __init__(self, in_features, out_features, activation="linear", kernel_initializer="he_uniform", bias_initializer="zeros"):
        self.weights = Tensor_(_Initializer(kernel_initializer)(in_features, out_features, shape=(in_features, out_features)), requires_grad=True)
        self.biases = Tensor_(_Initializer(bias_initializer)(in_features, out_features, shape=(out_features,)), requires_grad=True)
        self.activation = Activation(activation)

        self.params = [self.weights, self.biases]
    
    def __call__(self, x, training=True):
        self.output = matmul(x, self.weights) + self.biases
        self.output = self.activation(self.output, training=training)

        return self.output

    def parameters(self):
        return self.params

class BatchNormalization:
    def __init__(self, features=None, axis=0, momentum=0.99, gamma_initializer="ones", beta_initializer="zeros", moving_mean_initializer="zeros", moving_variance_initializer="ones"):
        self.axis = axis
        self.momentum = momentum

        self.gamma = Tensor_(_Initializer(gamma_initializer)(features, shape=(features,)), requires_grad=True)
        self.beta = Tensor_(_Initializer(beta_initializer)(features, shape=(features,)), requires_grad=True)

        self.running_mean = _Initializer(moving_mean_initializer)(features, shape=(features,))
        self.running_var = _Initializer(moving_variance_initializer)(features, shape=(features,))

        self.params = [self.gamma, self.beta]
    
    def __call__(self, x, training=True):
        if training:
            x_item = to_numpy(x)

            mean = np.mean(x_item, axis=self.axis)
            var = np.var(x_item, axis=self.axis)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self.output = self.gamma * ((x - mean) / (np.sqrt(var) + epsilon)) + self.beta
        return self.output

    def parameters(self):
        return self.params
