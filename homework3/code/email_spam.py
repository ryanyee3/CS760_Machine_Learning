import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix, log_loss, roc_curve

data = pd.read_csv("homework3/data/emails.csv")
X = data.iloc[:, 1:3001]
y = data.Prediction

def knn_cv(k, fold, X=X, y=y):
    start = (fold - 1) * 1000
    stop = (fold * 1000)
    X_train = X.drop(range(start, stop), axis=0)
    y_train = y.drop(range(start, stop), axis=0)
    X_test = X.iloc[start:stop, :]
    y_test = y.iloc[start:stop]
    model = knn(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(fold, " & ", round(accuracy, 4), " & ", round(precision, 4), " & ", round(recall, 4), " \\\\")
    return(accuracy)

def sigma(z):
    return(1 / (1 + np.exp(-z)))
    
def cross_entropy_loss(theta, X, y):
    return(-np.inner(y, np.log(sigma(X.dot(theta)))) - np.inner((1 - y), np.log(1 - sigma(X.dot(theta)))))

def log_reg(X, y, step_size, max_it=100, stop_delta=10**-4):
    p = len(X.iloc[0, ])
    n = len(y)
    theta = np.zeros(p)
    it = 1
    loss_old = log_loss(y, sigma(X.dot(theta)))
    delta = stop_delta
    while (it <= max_it) & (delta >= stop_delta):
        theta_old = theta
        theta = theta_old - step_size * np.sum((X.T * (sigma(X.dot(theta_old)) - y)).T, axis = 0) / n
        loss = log_loss(y, sigma(X.dot(theta)))
        delta = abs(loss_old - loss)
        loss_old = loss
        if it % 100 == 0:
            print("it:", it, "delta:", delta, "cur_loss:", loss)
        it += 1
    return(theta)

def cv_log_reg(fold, X, y, step_size, max_it=100, stop_delta=10**-4):
    start = (fold - 1) * 1000
    stop = (fold * 1000)
    X_train = X.drop(range(start, stop), axis=0)
    y_train = y.drop(range(start, stop), axis=0)
    X_test = X.iloc[start:stop, :]
    y_test = y.iloc[start:stop]
    theta = log_reg(X_train, y_train, step_size, max_it=max_it, stop_delta=stop_delta)
    probs = sigma(X_test.dot(theta))
    predictions = [1 if x >= .5 else 0 for x in probs]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print(fold, " & ", round(accuracy, 4), " & ", round(precision, 4), " & ", round(recall, 4), " \\\\")


if __name__ == '__main__':
    # for i in range(1, 6):
    #     knn_cv(1, i)

    # for i in range(1, 6):
    #     cv_log_reg(i, X, y, .003, max_it=1000)

    # k = [1, 3, 5, 7, 10]
    # folds = [1, 2, 3, 4, 5]
    # avg_acc = []
    # for i in k:
    #     a = [knn_cv(i, x) for x in folds]
    #     print(i, "&", round(sum(a) / 5, 4), "\\\\")
    #     avg_acc.append(sum(a) / 5)
    
    # plt.plot(k, avg_acc)
    # plt.scatter(k, avg_acc)
    # plt.title("kNN 5-Fold CV")
    # plt.xlabel("k")
    # plt.ylabel("Average Accuracy")
    # plt.savefig("homework3/figs/kNN_CV")

    # X_train = X.iloc[0:4000, ]
    # y_train = y.iloc[0:4000, ]
    # X_test = X.iloc[4000:5000, ]
    # y_test = y.iloc[4000:5000, ]
    # knn5= knn(n_neighbors=5)
    # knn5.fit(X_train, y_train)
    # knn_probs = knn5.predict_proba(X_test)
    # knn_pred = [i[1] for i in knn_probs]
    # fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, knn_pred)
    # log_reg_coef = log_reg(X_train, y_train, .003, max_it=1000)
    # log_reg_pred = sigma(X_test.dot(log_reg_coef))
    # fpr_log_reg, tpr_log_reg, thresh_log_reg = roc_curve(y_test, log_reg_pred)
    # plt.plot(fpr_knn, tpr_knn)
    # plt.plot(fpr_log_reg, tpr_log_reg)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.savefig("homework3/figs/spam_roc2.png")
    
    pass

