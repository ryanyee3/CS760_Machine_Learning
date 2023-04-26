import numpy as np
import torch
from torch.utils.data import DataLoader


class LinearSVM():

    def __init__(self, p):
        '''
        p   ---> number of parameters
        '''
        self.w = torch.rand(p, dtype=torch.double).requires_grad_(True)
        self.b = torch.rand(1, dtype=torch.double).requires_grad_(True)

    def loss(self, predictions, labels, C):
        '''
        predictions ---> 
        labels      ---> 
        C           ---> slack variable importance constant
        '''
        hinge_loss = sum(torch.max(torch.tensor([0.0]), 1 - labels * predictions))
        w_norm = (self.w**2).sum()
        return w_norm + C * hinge_loss
    
    def forward(self, data):
        '''
        data ---> assumes tensor with dtype=torch.double
        '''
        return data @ self.w - self.b
    
    def train(self, data, labels, C, epochs, step_size):
        '''
        data        ---> 
        labels      --->
        C           ---> slack variable importance constant
        epochs      --->
        step_size   --->
        '''
        X = torch.tensor(data, dtype=torch.double)
        y = torch.tensor(labels, dtype=torch.double)
        for i in range(epochs):
            predictions = self.forward(X)
            loss = self.loss(predictions, y, C)
            loss.backward()
            self.w.data -= step_size * self.w.grad
            self.b.data -= step_size * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()
            if i % 100 == 0:
                print(f"Epoch {i} of {epochs} ---> Loss: {loss:.8f}")
        return self.w, self.b


if __name__=='__main__':
    pass