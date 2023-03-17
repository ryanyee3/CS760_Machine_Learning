import numpy as np
from torchvision import datasets, transforms
from scipy.special import softmax, expit
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(
    root="homework4/data", 
    download=True, 
    train=True, 
    transform=transforms.ToTensor()
    )
mnist_test = datasets.MNIST(
    root="homework4/data", 
    download=True, 
    train=False, 
    transform=transforms.ToTensor()
    )

def get_inputs(data):
    X = np.zeros(len(data) * 28 * 28).reshape(len(data), 784)
    for i in range(len(data)):
        X[i] = np.array(data[i][0]).reshape(1, 784)[0]
    return(X.T)

def get_labels(data):
    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i][1]
    return(y)

X_train = get_inputs(mnist_train)
y_train = get_labels(mnist_train)
X_test = get_inputs(mnist_test)
y_test = get_labels(mnist_test)

def get_predictions(data, W1, W2, W3):
    return(softmax(np.matmul(W3, expit(np.matmul(W2, expit(np.matmul(W1, data)))))))

def get_loss(labels, predictions):
    loss = []
    for i in range(len(labels)):
        index = int(labels[i] - 1)
        y_hat = predictions[index, i]
        loss.append(np.log(y_hat))
    return(-sum(loss))

def get_one_hot_vectors(labels):
    mat = np.zeros((10 * len(labels))).reshape(10, len(labels))
    for i in range(len(labels)):
        index = int(labels[i] - 1)
        mat[index, i] = 1
    return(mat)

# initialize weights
W1 = np.random.uniform(low=0, high=1, size=(784 * 300)).reshape(300, 784)
W2 = np.random.uniform(low=0, high=1, size=(300 * 200)).reshape(200, 300)
W3 = np.random.uniform(low=0, high=1, size=(200 * 10)).reshape(10, 200)

# training
h = .000001
epochs = 10
it = 1
batch_size = 60
y = get_one_hot_vectors(y_train)

train_loss = [get_loss(y_train, get_predictions(X_train, W1, W2, W3)) / 60000]
test_loss = [get_loss(y_test, get_predictions(X_test, W1, W2, W3)) / 10000]
error = []

print(get_predictions(X_test[:, 0], W1, W2, W3))

while it <= epochs:
    batch_ids = np.random.choice(range(len(y_train)), size=60000, replace=False) # randint(low=0, high=60000, size=batch_size)
    batches = len(y_train) / batch_size
    for b in range(int(batches)):
        start = b * batch_size
        stop = start + batch_size
        ids = batch_ids[start:stop]
        X_batch = X_train[:, ids]
        y_batch = y[:, ids]
        a1 = expit(np.matmul(W1, X_batch))
        a2 = expit(np.matmul(W2, a1))
        g = get_predictions(X_batch, W1, W2, W3)
        d3 = (g - y_batch) # * a3 * (1 - a3)
        d2 = np.matmul(W3.T, d3) * a2 * (1 - a2)
        d1 = np.matmul(W2.T, d2) * a1 * (1 - a1)
        W3 = W3 - h * np.matmul(d3, a2.T)
        W2 = W2 - h * np.matmul(d2, a1.T)
        W1 = W1 - h * np.matmul(d1, X_batch.T)
    print(get_predictions(X_test[:, 0], W1, W2, W3))
    print(it, get_loss(y_train, get_predictions(X_train, W1, W2, W3)) / 60000)
    train_loss.append(get_loss(y_train, get_predictions(X_train, W1, W2, W3)) / 60000)
    test_loss.append(get_loss(y_test, get_predictions(X_test, W1, W2, W3)) / 10000)
    pred = get_predictions(X_test, W1, W2, W3)
    e = 0
    for i in range(len(y_test)):
        p = np.argmax(pred[:, i])
        if p != int(y_test[i]): e += 1
    error.append(e / len(y_test))
    it += 1

for i in range(len(error)):
    print(i+1, "&", round(error[i]*100, 4), "\%", "\\\\")

plt.plot(range(epochs + 1), train_loss, label='train loss')
plt.plot(range(epochs + 1), test_loss, label='test loss')
plt.legend(loc="upper right")
plt.title("Learning Curve")
plt.xlabel("epoch")
plt.ylabel("Average Loss")
plt.savefig("homework4/figs/learning_curve.png")

if __name__ == '__main__':
    pass
