import milligrad.alg as alg
import milligrad.backward as backward
import milligrad.data as data
import milligrad.layers as layers
import milligrad.losses as losses
import milligrad.models as models
import milligrad.optimizers as optimizers
import milligrad.utils as utils


def tensor(value, requires_grad=False):
    return backward.Tensor_(value, requires_grad=requires_grad)

class Tensor(backward.Tensor_):
    def __init__(self, value, requires_grad=False):
        super().__init__(value, requires_grad=requires_grad)

class no_grad(utils.no_grad_):
    def __init__(self):
        super().__init__()
