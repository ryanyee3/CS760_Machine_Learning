import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("homework3/data/Q5.txt", sep=" ", header=None, names=["y", "c"])

n_pos = sum(data.y)
n_neg = len(data.y) - n_pos

tp, fp, last_tp = (0, 0, 0)
roc_x = [0]
roc_y = [0]
for i in range(len(data.y)):
    if (i > 0):
        if (data.c[i] != data.c[i-1]) & (data.y[i] == 0) & (tp > last_tp):
            roc_x.append(fp / n_neg)
            roc_y.append(tp / n_pos)
            last_tp = tp
            print(data.c[i])
    if data.y[i] == 1:
        tp += 1
    else:
        fp += 1

roc_x.append(1)
roc_y.append(1)
plt.plot(roc_x, roc_y)
plt.scatter(roc_x, roc_y)
plt.title("ROC Curve")
plt.savefig("homework3/figs/q5_roc_curve.png")

if __name__ == '__main__':
    pass