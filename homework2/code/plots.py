import pandas as pd
from matplotlib import pyplot as plt
import tree as t
from plotnine import ggplot, aes, geom_point, ggsave, labs, geom_hline

q2 = pd.read_csv("homework2/data/Q2.txt", sep=" ", header=None, names=["x1", "x2", "y"])
d1 = pd.read_csv("homework2/data/D1.txt", sep=" ", header=None, names=["x1", "x2", "y"])
d2 = pd.read_csv("homework2/data/D2.txt", sep=" ", header=None, names=["x1", "x2", "y"])

q2_plot = ggplot(q2, aes("x1", "x2")) + geom_point(aes(color="y"))
d1_plot = ggplot(d1, aes("x1", "x2")) + geom_point(aes(color="y")) + labs(title="D1 Data") + geom_hline(aes(yintercept=.2))
d2_plot = ggplot(d2, aes("x1", "x2")) + geom_point(aes(color="y")) + labs(title="D2 Data")

def visualize_tree(data, x1_range, x2_range):
    tree = t.make_subtree(data)
    plt.scatter(data.x1, data.x2, c=data.y, cmap= 'viridis')
    draw_lines(tree, x1_range, x2_range)

def draw_lines(sub_tree, x1_range, x2_range):
    if sub_tree['type'] == 'leaf':
        pass
    else:
        c = sub_tree['cut_val']
        if sub_tree['cut_var'] == 0:
            plt.plot([c, c], x2_range)
            draw_lines(sub_tree["right_child"], [c, x1_range[1]], x2_range)
            draw_lines(sub_tree['left_child'], [x1_range[0], c], x2_range)
        else:
            plt.plot(x1_range, [c, c])
            draw_lines(sub_tree["right_child"], x1_range, [c, x2_range[1]])
            draw_lines(sub_tree['left_child'], x1_range, [x2_range[0], c])

if __name__ == "__main__":
    ggsave(filename="homework2/plots/q2_plot.png", plot=q2_plot)
    ggsave(filename="homework2/plots/d1_plot.png", plot=d1_plot)
    ggsave(filename="homework2/plots/d2_plot.png", plot=d2_plot)
    visualize_tree(d1, [0, 1], [0, 1])
    plt.savefig("homework2/plots/d1_tree_viz.png")
    plt.clf()
    visualize_tree(d2, [0, 1], [0, 1])  
    plt.savefig("homework2/plots/d2_tree_viz.png")
    pass
