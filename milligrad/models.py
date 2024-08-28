class Sequential:
    def __init__(self, layers, training=True):
        self.layers = layers
        self.training = training

        self.params = [param for layer in self.layers for param in layer.parameters()]
    
    def __call__(self, x):
        self.output = x
        for layer in self.layers:
            self.output = layer(self.output, training=self.training)
        
        return self.output
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def parameters(self):
        return self.params
