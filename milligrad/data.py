import numpy as np


class DataLoader:
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.x))
            self.x = self.x[indices]
            self.y = self.y[indices]

        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.x):
            raise StopIteration
        
        self.next_index = min(self.index + self.batch_size, len(self.x))
        
        x_batch = self.x[self.index:self.next_index]
        y_batch = self.y[self.index:self.next_index]

        self.index = self.next_index

        return x_batch, y_batch
