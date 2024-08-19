import numpy as np

epsilon = 1e-8
track_grad = True
e = np.e

class no_grad_:
    def __enter__(self):
        global track_grad
        self.prev_track_grad = track_grad
        track_grad = False
    
    def __exit__(self, exc_type, exc_value, traceback):
        global track_grad
        track_grad = self.prev_track_grad

def _ln(arr):
    return np.clip(np.log(arr), -100., None)

def getitem(arr):
    try:
        return np.array(arr.item(), dtype=np.float64)
    except (AttributeError, ValueError):
        return np.array(arr, dtype=np.float64)

def check_grad(arr, raise_exception=False):
    if not track_grad:
        if raise_exception:
            raise Exception("Gradient tracking is disabled globally")
        
        return False
    
    try:
        if not arr.requires_grad():
            if raise_exception:
                raise Exception("Gradient tracking is disabled for this Tensor")
            
            return False
    except AttributeError:
        if raise_exception:
            raise Exception("Not a Tensor object")
        
        return False

    return True

def check_item(arr, req, raise_exception=False):
    if not np.allclose(getitem(arr), getitem(req)):
        if raise_exception:
            raise Exception("Tensor item has changed after computation")
        
        return False

    return True

def check_shape(arr, req, raise_exception=False):
    if arr.shape != req.shape:
        if raise_exception:
            raise Exception("Shapes do not match")
        
        return False

    return True

def check_type(arr, reqs, raise_exception=False):
    if not isinstance(reqs, tuple): reqs = (reqs,)

    for req in reqs:
        if isinstance(arr, req):
            return True

    if raise_exception:
        raise Exception("Type does not match")
    
    return False

def check_leaf(arr, raise_exception=False):
    if arr._grad_fn:
        if raise_exception:
            raise Exception("Operation not allowed on non-leaf Tensor")
        
        return False

    return True

def package_grad(curr, req):
    curr, req = getitem(curr), getitem(req)

    if curr.shape == req.shape:
        return curr

    if curr.ndim >= req.ndim:
        dims_to_remove = np.arange(curr.ndim - req.ndim)
        removed = np.sum(curr, axis=tuple(dims_to_remove))

        dims_to_tighten = removed.ndim - req.ndim + np.where(np.array(removed.shape[-req.ndim:]) != np.array(req.shape))[0]
        tightened = np.sum(removed, axis=tuple(dims_to_tighten), keepdims=True)
        
        return tightened

    return np.broadcast_to(curr, req.shape).copy()

def _backward_checks(values, items):
    for key, val in values.items():
        check_grad(val, raise_exception=True)
        check_item(val, items[key], raise_exception=True)
    
    return True

def _sigmoid(arr):
    return np.piecewise(arr, [arr > 0.], [lambda x: 1 / (1 + np.exp(-x) + epsilon), lambda x: np.exp(x) / (1 + np.exp(x) + epsilon)])

def _softmax(arr, axis=-1):
    maximum_values = np.max(arr, axis=axis, keepdims=True)
    exponentials = np.exp(arr - maximum_values)

    return exponentials / (np.sum(exponentials, axis=axis, keepdims=True) + epsilon)

def _topo_sort_grad_fns(start_fn):
    visited = set()
    stack = []
    
    def dfs(curr_fn):
        visited.add(curr_fn)
        
        for node in curr_fn._grad_inputs.values():
            if node._grad_fn:
                if node._grad_fn not in visited:
                    dfs(node._grad_fn)
        
        stack.append(curr_fn)
    
    dfs(start_fn)
    return stack

def _transpose(arr):
    return np.swapaxes(arr, -1, -2)
