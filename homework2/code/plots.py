import pandas as pd
from plotnine import ggplot, aes, geom_point, ggsave, labs, geom_hline

q2 = pd.read_csv("homework2/data/Q2.txt", sep=" ", header=None, names=["x1", "x2", "y"])
d1 = pd.read_csv("homework2/data/D1.txt", sep=" ", header=None, names=["x1", "x2", "y"])
d2 = pd.read_csv("homework2/data/D2.txt", sep=" ", header=None, names=["x1", "x2", "y"])

q2_plot = ggplot(q2, aes("x1", "x2")) + geom_point(aes(color="y"))
d1_plot = ggplot(d1, aes("x1", "x2")) + geom_point(aes(color="y")) + labs(title="D1 Data") + geom_hline(aes(yintercept=.2))
d2_plot = ggplot(d2, aes("x1", "x2")) + geom_point(aes(color="y")) + labs(title="D2 Data")

if __name__ == "__main__":
    ggsave(filename="homework2/plots/q2_plot.png", plot=q2_plot)
    ggsave(filename="homework2/plots/d1_plot.png", plot=d1_plot)
    ggsave(filename="homework2/plots/d2_plot.png", plot=d2_plot)
