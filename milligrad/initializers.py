import numpy as np


def glorot_uniform(*args, **kwargs):
    in_features = args[0]
    out_features = args[1] if len(args) > 1 else in_features
    shape = kwargs["shape"]

    max_val = np.sqrt(6. / (in_features + out_features))
    return np.random.uniform(-max_val, max_val, shape)

def glorot_normal(*args, **kwargs):
    in_features = args[0]
    out_features = args[1] if len(args) > 1 else in_features
    shape = kwargs["shape"]

    std = np.sqrt(2. / (in_features + out_features))
    return np.random.normal(0., std, shape)

def he_uniform(*args, **kwargs):
    in_features = args[0]
    shape = kwargs["shape"]

    max_val = np.sqrt(6. / in_features)
    return np.random.uniform(-max_val, max_val, shape)

def he_normal(*args, **kwargs):
    in_features = args[0]
    shape = kwargs["shape"]

    std = np.sqrt(2. / in_features)
    return np.random.normal(0., std, shape)

def zeros(*_, **kwargs):
    shape = kwargs["shape"]
    return np.zeros(shape)

def ones(*_, **kwargs):
    shape = kwargs["shape"]
    return np.ones(shape)

class _Initializer:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        if self.name == "glorot_uniform":
            self.output = glorot_uniform(*args, **kwargs)
        elif self.name == "glorot_normal":
            self.output = glorot_normal(*args, **kwargs)
        elif self.name == "he_uniform": 
            self.output = he_uniform(*args, **kwargs)
        elif self.name == "he_normal":
            self.output = he_normal(*args, **kwargs)
        elif self.name == "zeros":
            self.output = zeros(*args, **kwargs)
        elif self.name == "ones":
            self.output = ones(*args, **kwargs)
        
        return self.output
