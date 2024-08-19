from .backward import (BCENode, BCEWithLogitsNode, BroadcastNode,
                       CrossEntropyNode, CrossEntropyWithLogitsNode, LogNode,
                       MatmulNode, MaxNode, MeanNode, MinNode, MSENode,
                       SigmoidNode, SoftmaxNode, SumNode, TanhNode)
from .utils import e


def log(antilogarithm, base=e):
    return LogNode(base, antilogarithm)._output

def matmul(arr1, arr2):
    return MatmulNode(arr1, arr2)._output

def maximum(arr1, arr2):
    return MaxNode(arr1, arr2)._output

def relu(arr):
    return maximum(arr, 0.)

def minimum(arr1, arr2):
    return MinNode(arr1, arr2)._output

def tanh(arr):
    return TanhNode(arr)._output

def sigmoid(arr):
    return SigmoidNode(arr)._output

def softmax(arr, axis=-1):
    return SoftmaxNode(arr, axis)._output

def binary_cross_entropy(y_true, y_pred, from_logits=False):
    if from_logits:
        return BCEWithLogitsNode(y_true, y_pred)._output
    
    return BCENode(y_true, y_pred)._output

def cross_entropy(y_true, y_pred, axis=-1, from_logits=False):
    if from_logits:
        return CrossEntropyWithLogitsNode(y_true, y_pred, axis)._output
    
    return CrossEntropyNode(y_true, y_pred, axis)._output

def mean_squared_error(y_true, y_pred):
    return MSENode(y_true, y_pred)._output

def mean(arr, axis=None, keepdims=False):
    return MeanNode(arr, axis, keepdims)._output

def sum(arr, axis=None, keepdims=False):
    return SumNode(arr, axis, keepdims)._output

def broadcast_to(arr, shape):
    return BroadcastNode(arr, shape)._output
