import numpy as np

from .utils import (_ln, _sigmoid, _softmax, _topological_sort, _transpose,
                    epsilon, track_grad)


def to_numpy(arr):
    if isinstance(arr, Tensor_):
        return arr.item()
    
    if not isinstance(arr, np.ndarray):
        return np.array(arr, dtype=np.float64)
    
    return arr

def package_grad(curr, req):
    curr, req = to_numpy(curr), to_numpy(req)

    if curr.shape == req.shape:
        return curr

    if curr.ndim >= req.ndim:
        dims_to_remove = range(curr.ndim - req.ndim)
        removed = np.add.reduce(curr, axis=tuple(dims_to_remove))

        dims_to_tighten = (i for i, (rem_dim, req_dim) in enumerate(zip(removed.shape, req.shape)) if rem_dim != req_dim)
        tightened = np.add.reduce(removed, axis=tuple(dims_to_tighten), keepdims=True)
        
        return tightened
    
    result = np.empty_like(req)
    result[...] = curr
    return result

def check_grad(arr, raise_exception=False):
    if not track_grad:
        if raise_exception:
            raise Exception("Gradient tracking is disabled globally")
        
        return False

    if not isinstance(arr, Tensor_):
        if raise_exception:
            raise Exception("Not a Tensor object")
        
        return False
    
    if not arr.requires_grad():
        if raise_exception:
            raise Exception("Gradient tracking is disabled for this Tensor")
            
        return False

    return True

def check_item(arr, req, raise_exception=False):
    if not np.allclose(arr, req):
        if raise_exception:
            raise Exception("Tensor item has changed after computation")
        
        return False

    return True

def check_type(arr, *reqs, raise_exception=False):
    for req in reqs:
        if isinstance(arr, req):
            return True

    if raise_exception:
        raise Exception("Type does not match")
    
    return False

def _backward_checks(grad_inputs, items):
    for key, grad_val in grad_inputs.items():
        check_grad(grad_val, raise_exception=True)
        check_item(grad_val.item(), items[key], raise_exception=True)
    
    return True

class Tensor_:
    def __init__(self, value, requires_grad=False):
        self._grad_fn = None
        self._item = np.array(value, dtype=np.float64)
        self._grad = np.zeros_like(self._item, dtype=np.float64)
        self._requires_grad = requires_grad
    
    def __neg__(self):
        return NegateNode(self)._output
    
    def __add__(self, other):
        return PlusNode(self, other)._output

    def __radd__(self, other):
        return PlusNode(other, self)._output
    
    def __sub__(self, other):
        return SubtractNode(self, other)._output
    
    def __rsub__(self, other):
        return SubtractNode(other, self)._output
    
    def __mul__(self, other):
        return MultiplyNode(self, other)._output

    def __rmul__(self, other):
        return MultiplyNode(other, self)._output
    
    def __truediv__(self, other):
        return DivideNode(self, other)._output
    
    def __rtruediv__(self, other):
        return DivideNode(other, self)._output
    
    def __pow__(self, other):
        return PowerNode(self, other)._output

    def __rpow__(self, other):
        return PowerNode(other, self)._output
    
    def __matmul__(self, other):
        return MatmulNode(self, other)._output
    
    def __rmatmul__(self, other):
        return MatmulNode(other, self)._output

    def item(self):
        return self._item
    
    def update_item(self, value):
        self._item += value
    
    def grad(self):
        return self._grad
    
    def set_grad(self, grad):
        self._grad = package_grad(grad, self._item)

    def update_grad(self, grad):
        self._grad += package_grad(grad, self._item)

    def zero_grad(self):
        self.set_grad(0.)

    def requires_grad(self):
        return self._requires_grad
    
    def requires_grad_(self, requires_grad=True):
        self._requires_grad = requires_grad
        if not requires_grad:
            self._grad_fn = None
    
    def detach(self):
        return Tensor_(self._item, requires_grad=False)

    def backward(self, check_graph=False, retain_graph=False):
        if self._item.ndim:
            raise Exception("Can only backpropagate from scalar outputs")
        
        if check_graph:
            check_grad(self, raise_exception=True)

            if self._grad_fn:
                check_item(self._item, self._grad_fn._items["output"], raise_exception=True)

        self.set_grad(1.)

        if self._grad_fn:
            graph = _topological_sort(self._grad_fn)

            for curr_func in reversed(graph):
                if check_graph:
                    _backward_checks(curr_func._grad_inputs, curr_func._items)
                
                curr_func._backward()

                if not retain_graph:
                    curr_func._output.requires_grad_(False)
                    del curr_func
    
    def __repr__(self):
        if not self._requires_grad:
            return f"mg.Tensor({self._item})"
        else:
            return f"mg.Tensor({self._item}, grad_fn={self._grad_fn.__class__.__name__})"

class _DeferCompute:
    def __init__(self, *funcs):
        self._funcs = funcs
        self._cache = {}

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._funcs[idx]()
        
        return self._cache[idx]

class Node:
    def __init__(self, raw_inputs, names):
        self._raw_inputs = raw_inputs
        self._names = names
        self._items = dict(zip(self._names, map(to_numpy, self._raw_inputs)))
    
    def _detect_gradients(self, output):
        self._grad_inputs = {}
        for i, name in enumerate(self._names):
            if check_grad(self._raw_inputs[i]):
                self._grad_inputs[name] = self._raw_inputs[i]

        self._output = output
        self._items["output"] = output.item()
        
        if self._grad_inputs:
            output.requires_grad_()
            output._grad_fn = self
    
    def _update_grads(self, grad_updates, force_grad=False):
        for i, name in enumerate(self._names):
            if name in self._grad_inputs:
                grad_update = grad_updates[i]
                if not force_grad:
                    grad_update = self._output.grad() * grad_update
                
                self._grad_inputs[name].update_grad(grad_update)

class UnaryNode(Node):
    def __init__(self, node):
        super().__init__([node], ["node"])

class BinaryNode(Node):
    def __init__(self, left, right):
        super().__init__([left, right], ["left", "right"])

class BroadcastNode(UnaryNode):
    def __init__(self, node, shape, perform_checks=False):
        if perform_checks:
            check_type(shape, tuple, raise_exception=True)

        super().__init__(node)

        result = np.empty(shape)
        result[...] = self._items["node"]
        self._detect_gradients(Tensor_(result))
    
    def _backward(self):
        self._update_grads([1.])

class NegateNode(UnaryNode):
    def __init__(self, node):
        super().__init__(node)
        self._detect_gradients(Tensor_(-self._items["node"]))
    
    def _backward(self):
        self._update_grads([-1.])

class PlusNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] + self._items["right"]))

    def _backward(self):
        self._update_grads([1., 1.])

class SubtractNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] - self._items["right"]))
    
    def _backward(self):
        self._update_grads([1., -1.])

class MultiplyNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] * self._items["right"]))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._items["right"],
                                         lambda: self._items["left"]))

class DivideNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] / (self._items["right"] + epsilon)))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: 1. / (self._items["right"] + epsilon),
                                         lambda: -self._items["left"] / (self._items["right"] ** 2 + epsilon)))

class PowerNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "log_left" : _ln(self._items["left"]),
        }

        self._detect_gradients(Tensor_(self._items["left"] ** self._items["right"]))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._items["right"] * (self._items["left"] ** (self._items["right"] - 1.)),
                                         lambda: self._items["output"] * self._extras["log_left"]))

class LogNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "log_left" : _ln(self._items["left"]),
            "log_right" : _ln(self._items["right"])
        }

        self._detect_gradients(Tensor_(self._extras["log_right"] / self._extras["log_left"]))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: -self._items["output"] / (self._items["left"] * self._extras["log_left"] + epsilon),
                                         lambda: 1./ (self._items["right"] * self._extras["log_left"] + epsilon)))
 
class MatmulNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] @ self._items["right"]))

    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._output.grad() @ _transpose(self._items["right"]),
                                         lambda: _transpose(self._items["left"]) @ self._output.grad()),
                           force_grad=True)

class MaxNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "left_larger" : self._items["left"] > self._items["right"],
            "right_larger" : self._items["right"] > self._items["left"]
        }

        self._detect_gradients(Tensor_(np.maximum(self._items["left"], self._items["right"])))

    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._extras["left_larger"],
                                         lambda: self._extras["right_larger"]))

class MinNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "left_lower" : self._items["left"] < self._items["right"],
            "right_lower" : self._items["right"] < self._items["left"]
        }

        self._detect_gradients(Tensor_(np.minimum(self._items["left"], self._items["right"])))

    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._extras["left_lower"],
                                         lambda: self._extras["right_lower"]))

class TanhNode(UnaryNode):
    def __init__(self, node):
        super().__init__(node)
        self._detect_gradients(Tensor_(np.tanh(self._items["node"])))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: 1. - self._items["output"] ** 2))

class SigmoidNode(UnaryNode):
    def __init__(self, node):
        super().__init__(node)
        self._detect_gradients(Tensor_(_sigmoid(self._items["node"])))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._items["output"] * (1 - self._items["output"])))

class SoftmaxNode(UnaryNode):
    def __init__(self, node, axis=-1, perform_checks=False):
        if perform_checks:
            check_type(axis, int, raise_exception=True)

        super().__init__(node)

        self._extras = {
            "axis" : axis
        }

        self._detect_gradients(Tensor_(_softmax(self._items["node"], axis=axis)))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._items["output"] * (self._output.grad() - np.add.reduce(self._output.grad() * self._items["output"], axis=self._extras["axis"], keepdims=True))),
                           force_grad=True)

class BCENode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "log_right" : _ln(self._items["right"]),
            "log_flip_right" : _ln(1. - self._items["right"])
        }

        self._detect_gradients(Tensor_(-(self._items["left"] * self._extras["log_right"] + (1. - self._items["left"]) * self._extras["log_flip_right"])))

    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._extras["log_flip_right"] - self._extras["log_right"],
                                         lambda: (1. - self._items["left"]) / (1. - self._items["right"] + epsilon) - self._items["left"] / (self._items["right"] + epsilon)))

class BCEWithLogitsNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {}
        self._extras["sigmoid_right"] = _sigmoid(self._items["right"])
        self._extras["log_sigmoid_right"] = _ln(self._extras["sigmoid_right"])
        self._extras["log_flip_sigmoid_right"] = _ln(1. - self._extras["sigmoid_right"])

        self._detect_gradients(Tensor_(-(self._items["left"] * self._extras["log_sigmoid_right"] + (1. - self._items["left"]) * self._extras["log_flip_sigmoid_right"])))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: self._extras["log_flip_sigmoid_right"] - self._extras["log_sigmoid_right"],
                                         lambda: self._extras["sigmoid_right"] - self._items["left"]))

class CrossEntropyNode(BinaryNode):
    def __init__(self, left, right, axis=-1, perform_checks=False):
        if perform_checks:
            check_type(axis, int, raise_exception=True)

        super().__init__(left, right)

        self._extras = {
            "axis" : axis,
            "log_right" : _ln(self._items["right"]),
        }

        self._detect_gradients(Tensor_(-np.add.reduce(self._items["left"] * self._extras["log_right"], axis=axis)))

    def _backward(self):
        self._update_grads(_DeferCompute(lambda: -np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * self._extras["log_right"],
                                         lambda: -np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * (self._items["left"] / (self._items["right"] + epsilon))),
                           force_grad=True)

class CrossEntropyWithLogitsNode(BinaryNode):
    def __init__(self, left, right, axis=-1, perform_checks=False):
        if perform_checks:
            check_type(axis, int, raise_exception=True)

        super().__init__(left, right)

        self._extras = {"axis" : axis}
        self._extras["softmax_right"] = _softmax(self._items["right"], axis=axis)
        self._extras["log_softmax_right"] = _ln(self._extras["softmax_right"])

        self._detect_gradients(Tensor_(-np.add.reduce(self._items["left"] * self._extras["log_softmax_right"], axis=axis)))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: -np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * self._extras["log_softmax_right"],
                                         lambda: np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * (self._extras["softmax_right"] - self._items["left"])),
                           force_grad=True)

class MSENode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(((self._items["left"] - self._items["right"]) ** 2)))
    
    def _backward(self):
        self._update_grads(_DeferCompute(lambda: 2 * (self._items["left"] - self._items["right"]),
                                         lambda: 2 * (self._items["right"] - self._items["left"])))

class MeanNode(UnaryNode):
    def __init__(self, node, axis=None, keepdims=False, perform_checks=False):
        if perform_checks:
            check_type(axis, type(None), int, tuple, list, raise_exception=True)

            if axis and not isinstance(axis, int):
                for ax in axis:
                    check_type(ax, int, raise_exception=True)
                
                axis = tuple(axis)

            check_type(keepdims, bool, raise_exception=True)

        super().__init__(node)

        self._extras = {
            "axis" : (axis,) if isinstance(axis, int) else axis,
            "keepdims" : keepdims,
        }

        self._extras["axes_lost"] = range(self._items["node"].ndim) if axis is None else self._extras["axis"]
        self._extras["num_lost_elements"] = np.multiply.reduce([self._items["node"].shape[ax] for ax in self._extras["axes_lost"]])

        self._detect_gradients(Tensor_(np.add.reduce(self._items["node"], axis=axis, keepdims=keepdims) / self._extras["num_lost_elements"]))
    
    def _backward(self):
        grad_update = _DeferCompute(lambda: (self._output.grad() / self._extras["num_lost_elements"] if self._extras["keepdims"]
                                             else np.expand_dims(self._output.grad() / self._extras["num_lost_elements"], axis=tuple(self._extras["axes_lost"]))))
        
        self._update_grads(grad_update, force_grad=True)

class SumNode(UnaryNode):
    def __init__(self, node, axis=None, keepdims=False, perform_checks=False):
        if perform_checks:
            check_type(axis, type(None), int, tuple, list, raise_exception=True)

            if axis and not isinstance(axis, int):
                for ax in axis:
                    check_type(ax, int, raise_exception=True)
                
                axis = tuple(axis)

            check_type(keepdims, bool, raise_exception=True)

        super().__init__(node)

        self._extras = {
            "axis" : (axis,) if isinstance(axis, int) else axis,
            "keepdims" : keepdims,
            "axes_lost" : range(self._items["node"].ndim) if axis is None else axis
        }

        self._detect_gradients(Tensor_(np.add.reduce(self._items["node"], axis=axis, keepdims=keepdims)))
    
    def _backward(self):
        grad_update = _DeferCompute(lambda: (self._output.grad() if self._extras["keepdims"]
                                             else np.expand_dims(self._output.grad(), axis=tuple(self._extras["axes_lost"]))))
        
        self._update_grads(grad_update, force_grad=True)
