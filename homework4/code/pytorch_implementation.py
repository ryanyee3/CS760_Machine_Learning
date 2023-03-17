import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(
    root="homework4/data", 
    download=True, 
    train=True, 
    transform=transforms.ToTensor(), 
    target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )
mnist_test = datasets.MNIST(
    root="homework4/data", 
    download=True, 
    train=False, 
    transform=transforms.ToTensor(),
    target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.Sigmoid(),
            nn.Linear(300, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    plt_train_loss.append(loss.item())
    
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y.argmax(dim=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    plt_test_loss.append(test_loss)
    correct /= size
    test_errors.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Avg loss: {test_loss:>8f} \n")

# hyperparameters
learning_rate = .01
batch_size = 64
epochs = 20

train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

plt_train_loss = []
plt_test_loss = []

test_errors = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

for i in range(epochs):
    print(i+1, "&", round((1 - test_errors[i])*100, 4), "\%", "\\\\")

plt.plot(range(len(plt_train_loss)), plt_train_loss, label='train loss')
plt.plot(range(len(plt_test_loss)), plt_test_loss, label='test loss')
plt.legend(loc="upper right")
plt.title("Learning Curve")
plt.xlabel("epoch")
plt.ylabel("Average Loss")
plt.savefig("homework4/figs/pytorch_learning_curve.png")

if __name__ == '__main__':
    pass
