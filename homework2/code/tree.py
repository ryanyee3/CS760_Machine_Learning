import numpy as np
import pandas as pd
import math
import os, sys

# d1 = open("homework2/data/D1.txt", "r").read()
d1 = pd.read_csv("homework2/data/D1.txt", sep=" ", header=None, names=["x1", "x2", "y"])

def make_subtree(data):
    pass

def get_info_entropy(data):
    '''
    returns information entropy of the data [i.e. H(Y)]
    '''
    n1 = sum(data.y)
    n0 = len(data.y) - n1
    p1 = n1 / (n1 + n0)
    p0 = n0 / (n1 + n0)
    return -p0 * math.log2(p0) - p1 * math.log2(p1)

def get_info_entropy_gain(data, j, c):
    '''
    returns the entropy gain from splitting on variable j at point c
    '''
    hy = get_info_entropy(data)
    if j == 1:
        hy_less = get_info_entropy(data[data.x1 < c])
        p_less = len(data[data.x1 < c]) / len(data)
        hy_gtet = get_info_entropy(data[data.x1 >= c])
        p_gtet = len(data[data.x1 >= c]) / len(data)
    else:
        hy_less = get_info_entropy(data[data.x2 < c])
        p_less = len(data[data.x2 < c]) / len(data)
        hy_gtet = get_info_entropy(data[data.x2 >= c])
        p_gtet = len(data[data.x2 >= c]) / len(data)
    hy_x = p_less * hy_less + p_gtet * hy_gtet
    return hy - hy_x

def get_candidate_splints(data):
    pass

def find_best_split(data, candidates_splits):
    pass

if __name__=="__main__":
    print(get_info_entropy_gain(d1, 1, .8))
