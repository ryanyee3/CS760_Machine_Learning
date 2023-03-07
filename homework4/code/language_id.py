import numpy as np
import pandas as pd

# character types
S = [chr(i) for i in range(ord('a'), ord('z') + 1)]
S.append(" ")

# labels
L = ['e', 'j', 's']

# get character counts
def get_counts(label, characters, files):
    counts = np.zeros(len(characters))
    for i in range(len(files)):
        file = open("homework4/data/languageID/" + label + str(files[i]) + ".txt")
        data = file.read()
        for c in range(len(characters)):
            counts[c] = counts[c] + data.count(characters[c])
    return(counts)

# get conditional probabilities
def get_probabilities(counts, characters):
    N = sum(counts)
    probs = []
    for i in range(len(characters)):
        p = (counts[i] + 0.5) / (N + 27 * 0.5)
        probs.append(p)
    return(probs)

# print conditional probabilities for latex table
def print_probabilities(probabilities, characters):
    for i in range(len(characters)):
        print(characters[i], "&", round(probabilities[i], 4), "\\\\")

# print probabilities for each language
for l in range(len(L)):
    print(L[l])
    c = get_counts(L[l], S, range(10))
    p = get_probabilities(c, S)
    print_probabilities(p, S)

if __name__=='__main__':
    pass
