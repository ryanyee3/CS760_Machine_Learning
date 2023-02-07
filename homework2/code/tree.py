import numpy as np
import pandas as pd
import math

d1 = pd.read_csv("homework2/data/D1.txt", sep=" ", header=None, names=["x1", "x2", "y"])

def make_subtree(data):
    C = get_candidate_splits(data)
    if len(data) == 0:
        pass # make leaf
    elif len(C) == 0:
        pass # make leaf
    else:
        S = find_best_split(data, C)
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


def get_info_entropy_gain(data, c, j):
    '''
    returns the entropy gain from splitting at cut-point c on variable j
    '''
    hy = get_info_entropy(data)
    hy_less = get_info_entropy(data[data.iloc[:, j] < c])
    p_less = len(data[data.iloc[:, j] < c]) / len(data)
    hy_gtet = get_info_entropy(data[data.iloc[:, j] >= c])
    p_gtet = len(data[data.iloc[:, j] >= c]) / len(data)
    hy_x = p_less * hy_less + p_gtet * hy_gtet
    return hy - hy_x


def get_gain_ratio(data, c, j):
    '''
    returns entropy gain ratio from splitting at cut-point c on variable j
    '''
    p_less = len(data[data.iloc[:, j] < c]) / len(data)
    p_gtet = len(data[data.iloc[:, j] >= c]) / len(data)
    return get_info_entropy_gain(data, c, j) / (-p_less * math.log2(p_less) - p_gtet * math.log2(p_gtet))


def get_one_var_splits(data, j):
    '''
    returns splits for variable j
    '''
    C = []
    for i in range(len(data.iloc[:, j])):
        try:
            entropy_gain = get_info_entropy_gain(data, data.iloc[:, j][i], j)
            gain_ratio = get_gain_ratio(data, data.iloc[:, j][i], j)
        except:
            pass
        else:
            if entropy_gain > 0 & gain_ratio > 0:
                C.append([data.iloc[:, j][i], j])
    return C


def get_candidate_splits(data):
    '''
    returns 2D array of candidate splits c and corresponding variable index j
    '''
    return get_one_var_splits(data, 0) + get_one_var_splits(data, 1)


def find_best_split(data, candidates_splits):
    max_gain = 0
    max_index = None
    for i in range(len(candidates_splits)):
        try:
            cur_gain = get_info_entropy_gain(data, candidates_splits[i][0], candidates_splits[i][1])
        except:
            pass
        else:
            if cur_gain > max_gain:
                max_gain = cur_gain
                max_index = i
    return candidates_splits[max_index]


if __name__=="__main__":
    print(len(get_candidate_splits(d1)))

