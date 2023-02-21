import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt

data = pd.read_csv("homework3/data/D2z.txt", sep=" ", header=None, names=["x1", "x2", "y"])

X_train = data.iloc[:, 0:2]
y_train = data.y
knn1 = knn(n_neighbors=1)
knn1.fit(X_train, y_train)

test_points =[(x1, x2) for x1 in np.arange(-2, 2, .1) for x2 in np.arange(-2, 2, .1)]
test_predictions = knn1.predict(test_points)

x1_test = [i[0] for i in test_points]
x2_test = [i[1] for i in test_points]
plt.scatter(x1_test, x2_test, c=test_predictions, s=3)
plt.scatter(X_train.x1, X_train.x2, c=y_train, marker="*")
plt.savefig("homework3/figs/knn_d2z_plot.png")

if __name__ == '__main__':
    pass

