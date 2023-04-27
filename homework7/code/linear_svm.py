import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


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
    
    def train(self, data, labels, C, epochs, step_size, verbose=True):
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

            # forward pass
            predictions = self.forward(X)

            # backward pass
            loss = self.loss(predictions, y, C)
            loss.backward()

            # update parameters
            self.w.data -= step_size * self.w.grad
            self.b.data -= step_size * self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()

            # report results
            if verbose & (i % 100 == 0):
                print(f"Epoch {i} of {epochs} ---> Loss: {loss:.8f}")
        
        return self.w, self.b
    
    def get_accuracy(self, data, labels):
        X = torch.tensor(data, dtype=torch.double)
        pred_labels = torch.where(self.forward(X) < 0, -1, 1)
        return accuracy_score(pred_labels, labels)


if __name__=='__main__':
    pass