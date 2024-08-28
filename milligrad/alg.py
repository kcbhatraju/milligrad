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

def softmax(arr, axis=-1, perform_checks=False):
    return SoftmaxNode(arr, axis, perform_checks)._output

def binary_cross_entropy(y_true, y_pred, from_logits=False):
    if from_logits:
        return BCEWithLogitsNode(y_true, y_pred)._output
    
    return BCENode(y_true, y_pred)._output

def cross_entropy(y_true, y_pred, axis=-1, from_logits=False, perform_checks=False):
    if from_logits:
        return CrossEntropyWithLogitsNode(y_true, y_pred, axis, perform_checks)._output
    
    return CrossEntropyNode(y_true, y_pred, axis, perform_checks)._output

def mean_squared_error(y_true, y_pred):
    return MSENode(y_true, y_pred)._output

def broadcast_to(arr, shape, perform_checks=False):
    return BroadcastNode(arr, shape, perform_checks)._output

def mean(arr, axis=None, keepdims=False, perform_checks=False):
    return MeanNode(arr, axis, keepdims, perform_checks)._output

def sum(arr, axis=None, keepdims=False, perform_checks=False):
    return SumNode(arr, axis, keepdims, perform_checks)._output
