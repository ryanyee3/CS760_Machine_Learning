import pandas as pd
import tree as t
from matplotlib import pyplot as plt
from plots import visualize_tree
from plotnine import ggplot, aes, geom_point, geom_line, ggsave, labs, scale_x_log10
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("homework2/data/Dbig.txt", sep=" ", header=None, names=["x1", "x2", "y"]).sample(frac=1)

train = data.iloc[0:8192, :]
test = data.iloc[8192:10000, :]

def get_error(tree, test):
    n_err = 0
    for i in range(len(test)):
        true_val = test.iloc[i, 2]
        predicted_val = t.predict(tree, test.iloc[i, 0], test.iloc[i, 1])
        if true_val != predicted_val:
            n_err += 1
    return n_err / len(test)

def count_nodes(sub_tree):
    n = 0
    if sub_tree['type'] == 'leaf':
        pass
    else:
        n = 1 + count_nodes(sub_tree['right_child']) + count_nodes(sub_tree['left_child'])
    return n

def get_sklearn_error(model, test):
    n_err = 0
    predictions = model.predict(test.iloc[:, 0:2])
    for i in range(len(test)):
        if predictions[i] == test.iloc[i, 2]:
            pass
        else:
            n_err += 1
    return n_err / len(test)

if __name__ == "__main__":
    n = [32, 128, 512, 2048, 8192]

    trees = []
    error = []
    for i in range(len(n)):
        trees.append(t.make_subtree(train.iloc[0:n[i], :]))
        error.append(get_error(trees[i], test))
    
    for i in range(0, 5):
        print(n[i], " & ", count_nodes(trees[i]), " & ", round(error[i], 4), " \\\\")
        visualize_tree(train.iloc[0:n[i], :], [-1.5, 1.5], [-1.5, 1.5])
        plt.savefig("homework2/plots/dbig_" + str(n[i]) + ".png")
        plt.clf()

    error_data = pd.DataFrame(data = {
        "n": n, 
        "error": error})  
    error_curve = ggplot(error_data, aes(x="n", y="error")) + geom_point() + geom_line() + scale_x_log10() + labs(title="Learning Curve")
    ggsave(filename="homework2/plots/error_curve.png", plot=error_curve)

    sklearn_trees = []
    sklearn_error = []
    for i in range(len(n)):
        sklearn_trees.append(DecisionTreeClassifier().fit(train.iloc[0:n[i], 0:2], train.iloc[0:n[i], 2]))
        sklearn_error.append(get_sklearn_error(sklearn_trees[i], test))

    for i in range(0, 5):
        print(n[i], " & ", sklearn_trees[i].tree_.node_count, " & ", round(sklearn_error[i], 4), " \\\\")

    sklearn_error_data = pd.DataFrame(data = {
        "n": n, 
        "error": sklearn_error})
    sklearn_error_curve = ggplot(sklearn_error_data, aes(x="n", y="error")) + geom_point() + geom_line() + scale_x_log10() + labs(title="Learning Curve")
    ggsave(filename="homework2/plots/sklearn_error_curve.png", plot=sklearn_error_curve)

    pass

