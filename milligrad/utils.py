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

def _sigmoid(arr):
    return np.piecewise(arr, [arr > 0.], [lambda x: 1 / (1 + np.exp(-x) + epsilon), lambda x: np.exp(x) / (1 + np.exp(x) + epsilon)])

def _softmax(arr, axis=-1):
    maximum_values = np.max(arr, axis=axis, keepdims=True)
    exponentials = np.exp(arr - maximum_values)

    return exponentials / (np.add.reduce(exponentials, axis=axis, keepdims=True) + epsilon)

def _topological_sort(start_fn):
    visited = set()
    ordering = []
    
    def dfs(curr_fn):
        visited.add(curr_fn)
        
        for node in curr_fn._grad_inputs.values():
            if node._grad_fn:
                if node._grad_fn not in visited:
                    dfs(node._grad_fn)
        
        ordering.append(curr_fn)
    
    dfs(start_fn)
    return ordering

def _transpose(arr):
    return np.swapaxes(arr, -1, -2)
