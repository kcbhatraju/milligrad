import numpy as np

from .utils import (_backward_checks, _ln, _sigmoid, _softmax,
                    _topo_sort_grad_fns, _transpose, check_grad, check_item,
                    check_leaf, check_type, epsilon, getitem, package_grad)


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
    
    def set_item(self, value):
        check_leaf(self, raise_exception=True)
        self._item = getitem(value)
    
    def update_item(self, value):
        check_leaf(self, raise_exception=True)
        self._item += getitem(value)
    
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
        check_leaf(self, raise_exception=True)

        if requires_grad != self._requires_grad:
            self._requires_grad = requires_grad
            self._grad = np.zeros_like(self._item, dtype=np.float64)
    
    def detach(self):
        return Tensor_(self._item, requires_grad=False)

    def backward(self, retain_graph=False):
        if self._item.ndim > 0:
            raise Exception("Can only backpropagate from scalar outputs")
        
        check_grad(self, raise_exception=True)

        if self._grad_fn:
            check_item(self._item, self._grad_fn._items["output"], raise_exception=True)

        self.set_grad(1.)

        if self._grad_fn:
            graph = _topo_sort_grad_fns(self._grad_fn)

            while graph:
                curr_func = graph.pop()
                
                _backward_checks(curr_func._grad_inputs, curr_func._items)
                curr_func._backward()
                
                if not retain_graph:
                    curr_func._output._grad_fn = None
                    del curr_func
    
    def __repr__(self):
        if not self._requires_grad:
            return f"mg.Tensor({self._item})"
        else:
            return f"mg.Tensor({self._item}, grad_fn={self._grad_fn.__class__.__name__})"

class Node:
    def __init__(self, raw_inputs, names):
        self._raw_inputs = raw_inputs
        self._names = names
        self._populate_items()

    def _populate_items(self):
        self._items = {}
        for i, name in enumerate(self._names):
            self._items[name] = getitem(self._raw_inputs[i])
    
    def _detect_gradients(self, output):
        self._output = output
        at_least_one_grad = False

        self._grad_inputs = {}
        for i, name in enumerate(self._items.keys()):
            if check_grad(self._raw_inputs[i]):
                at_least_one_grad = True
                self._grad_inputs[name] = self._raw_inputs[i]
        
        self._items["output"] = output.item()

        if at_least_one_grad:
            output.requires_grad_()
            output._grad_fn = self
    
    def _update_grads(self, grad_updates, force_grad=False):
        for i, name in enumerate(self._items.keys()):
            if i >= len(grad_updates): break

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

class NegateNode(UnaryNode):
    def __init__(self, node):
        super().__init__(node)
        self._detect_gradients(Tensor_(-self._items["node"]))
    
    def _backward(self):
        if "node" in self._inputs:
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
        self._update_grads([self._items["right"],
                            self._items["left"]])

class DivideNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] / (self._items["right"] + epsilon)))
    
    def _backward(self):
        self._update_grads([1. / (self._items["right"] + epsilon),
                            -self._items["left"] / (self._items["right"] ** 2 + epsilon)])

class PowerNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "log_left" : _ln(self._items["left"]),
        }

        self._detect_gradients(Tensor_(self._items["left"] ** self._items["right"]))
    
    def _backward(self):
        self._update_grads([self._items["right"] * (self._items["left"] ** (self._items["right"] - 1.)), 
                            self._items["output"] * self._extras["log_left"]])

class LogNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "log_left" : _ln(self._items["left"]),
            "log_right" : _ln(self._items["right"])
        }

        self._detect_gradients(Tensor_(self._extras["log_right"] / self._extras["log_left"]))
    
    def _backward(self):
        self._update_grads([-self._items["output"] / (self._items["left"] * self._extras["log_left"] + epsilon),
                            1./ (self._items["right"] * self._extras["log_left"] + epsilon)])
 
class MatmulNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(self._items["left"] @ self._items["right"]))
    
    def _backward(self):
        self._update_grads([self._output.grad() @ _transpose(self._items["right"]),
                            _transpose(self._items["left"]) @ self._output.grad()],
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
        self._update_grads([self._extras["left_larger"],
                            self._extras["right_larger"]])

class MinNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {
            "left_lower" : self._items["left"] < self._items["right"],
            "right_lower" : self._items["right"] < self._items["left"]
        }

        self._detect_gradients(Tensor_(np.minimum(self._items["left"], self._items["right"])))

    def _backward(self):
        self._update_grads([self._extras["left_lower"],
                            self._extras["right_lower"]])

class TanhNode(UnaryNode):
    def __init__(self, node):
        super().__init__(node)
        self._detect_gradients(Tensor_(np.tanh(self._items["node"])))
    
    def _backward(self):
        self._update_grads([1. - self._items["output"] ** 2])

class SigmoidNode(UnaryNode):
    def __init__(self, node):
        super().__init__(node)
        self._detect_gradients(Tensor_(_sigmoid(self._items["node"])))
    
    def _backward(self):
        self._update_grads([self._items["output"] * (1 - self._items["output"])])

class SoftmaxNode(UnaryNode):
    def __init__(self, node, axis=-1):
        check_type(axis, int, raise_exception=True)

        super().__init__(node)

        self._extras = {
            "axis" : axis
        }

        self._detect_gradients(Tensor_(_softmax(self._items["node"], axis=axis)))
    
    def _backward(self):
        self._update_grads([self._items["output"] * (self._output.grad() - np.sum(self._output.grad() * self._items["output"], axis=self._extras["axis"], keepdims=True))],
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
        self._update_grads([self._extras["log_flip_right"] - self._extras["log_right"],
                            (1. - self._items["left"]) / (1. - self._items["right"] + epsilon) - self._items["left"] / (self._items["right"] + epsilon)])

class BCEWithLogitsNode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)

        self._extras = {}
        self._extras["sigmoid_right"] = _sigmoid(self._items["right"])
        self._extras["log_sigmoid_right"] = _ln(self._extras["sigmoid_right"])
        self._extras["log_flip_sigmoid_right"] = _ln(1. - self._extras["sigmoid_right"])

        self._detect_gradients(Tensor_(-(self._items["left"] * self._extras["log_sigmoid_right"] + (1. - self._items["left"]) * self._extras["log_flip_sigmoid_right"])))
    
    def _backward(self):
        self._update_grads([self._extras["log_flip_sigmoid_right"] - self._extras["log_sigmoid_right"],
                            self._extras["sigmoid_right"] - self._items["left"]])

class CrossEntropyNode(BinaryNode):
    def __init__(self, left, right, axis=-1):
        check_type(axis, int, raise_exception=True)

        super().__init__(left, right)

        self._extras = {
            "axis" : axis,
            "log_right" : _ln(self._items["right"]),
        }

        self._detect_gradients(Tensor_(-np.sum(self._items["left"] * self._extras["log_right"], axis=axis)))

    def _backward(self):
        self._update_grads([-np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * self._extras["log_right"],
                            -np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * (self._items["left"] / (self._items["right"] + epsilon))],
                            force_grad=True)

class CrossEntropyWithLogitsNode(BinaryNode):
    def __init__(self, left, right, axis=-1):
        check_type(axis, int, raise_exception=True)

        super().__init__(left, right)

        self._extras = {"axis" : axis}
        self._extras["softmax_right"] = _softmax(self._items["right"], axis=axis)
        self._extras["log_softmax_right"] = _ln(self._extras["softmax_right"])

        self._detect_gradients(Tensor_(-np.sum(self._items["left"] * self._extras["log_softmax_right"], axis=axis)))
    
    def _backward(self):
        self._update_grads([-np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * self._extras["log_softmax_right"],
                            np.expand_dims(self._output.grad(), axis=self._extras["axis"]) * (self._extras["softmax_right"] - self._items["left"])],
                            force_grad=True)

class MSENode(BinaryNode):
    def __init__(self, left, right):
        super().__init__(left, right)
        self._detect_gradients(Tensor_(((self._items["left"] - self._items["right"]) ** 2)))
    
    def _backward(self):
        self._update_grads([2 * (self._items["left"] - self._items["right"]),
                            2 * (self._items["right"] - self._items["left"])])

class MeanNode(UnaryNode):
    def __init__(self, node, axis=None, keepdims=False):
        check_type(axis, (type(None), int, tuple, list), raise_exception=True)
        if axis and not isinstance(axis, int):
            for ax in axis: check_type(ax, int, raise_exception=True)
            axis = tuple(axis)

        check_type(keepdims, bool, raise_exception=True)

        super().__init__(node)

        self._extras = {
            "axis" : (axis,) if isinstance(axis, int) else axis,
            "keepdims" : keepdims,
        }

        self._extras["axes_lost"] = tuple(range(self._items["node"].ndim)) if axis is None else self._extras["axis"]
        self._extras["num_lost_elements"] = np.prod([self._items["node"].shape[i] for i in self._extras["axes_lost"]])

        self._detect_gradients(Tensor_(np.mean(self._items["node"], axis=axis, keepdims=keepdims)))
    
    def _backward(self):
        grad_update = self._output.grad() / self._extras["num_lost_elements"]
        if not self._extras["keepdims"]:
            grad_update = np.expand_dims(grad_update, axis=self._extras["axes_lost"])
        
        self._update_grads([grad_update], force_grad=True)

class SumNode(UnaryNode):
    def __init__(self, node, axis=None, keepdims=False):
        check_type(axis, (type(None), int, tuple, list), raise_exception=True)
        if axis and not isinstance(axis, int):
            for ax in axis: check_type(ax, int, raise_exception=True)
            axis = tuple(axis)

        check_type(keepdims, bool, raise_exception=True)

        super().__init__(node)

        self._extras = {
            "axis" : (axis,) if isinstance(axis, int) else axis,
            "keepdims" : keepdims,
            "axes_lost" : tuple(range(self._items["node"].ndim)) if axis is None else axis
        }

        self._detect_gradients(Tensor_(np.sum(self._items["node"], axis=axis, keepdims=keepdims)))
    
    def _backward(self):
        grad_update = self._output.grad()
        if not self._extras["keepdims"]:
            grad_update = np.expand_dims(grad_update, axis=self._extras["axes_lost"])
        
        self._update_grads([grad_update], force_grad=True)

class BroadcastNode(UnaryNode):
    def __init__(self, node, shape):
        check_type(shape, tuple, raise_exception=True)

        super().__init__(node)
        self._detect_gradients(Tensor_(np.broadcast_to(self._items["node"], shape).copy()))
    
    def _backward(self):
        self._update_grads([1.])
