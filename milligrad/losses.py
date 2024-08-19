from .alg import (binary_cross_entropy, cross_entropy, mean,
                  mean_squared_error, sum)


class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
    
    def __call__(self, y_true, y_pred):
        loss = mean_squared_error(y_true, y_pred)
        if self.reduction == "mean":
            return mean(loss)
        elif self.reduction == "sum":
            return sum(loss)

class BCELoss:
    def __init__(self, reduction="mean", from_logits=False):
        self.reduction = reduction
        self.from_logits = from_logits
    
    def __call__(self, y_true, y_pred):
        loss = binary_cross_entropy(y_true, y_pred, from_logits=self.from_logits)
        if self.reduction == "mean":
            return mean(loss)
        elif self.reduction == "sum":
            return sum(loss)

class CrossentropyLoss:
    def __init__(self, reduction="mean", from_logits=False):
        self.reduction = reduction
        self.from_logits = from_logits
    
    def __call__(self, y_true, y_pred):
        loss = cross_entropy(y_true, y_pred, from_logits=self.from_logits)
        if self.reduction == "mean":
            return mean(loss)
        elif self.reduction == "sum":
            return sum(loss)
