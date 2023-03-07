import numpy as np
import pandas as pd
import math

# character types
S = [chr(i) for i in range(ord('a'), ord('z') + 1)]
S.append(" ")

# labels
L = ['e', 'j', 's']

class NaiveBayes:

    def __init__(self, label, characters, files):
        self.label = label
        self.characters = characters
        self.files = files
        self.counts = self.get_counts()
        self.probabilities = self.get_probabilities()

    def get_counts(self):
        counts = np.zeros(len(self.characters))
        for i in range(len(self.files)):
            file = open("homework4/data/languageID/" + self.label + str(self.files[i]) + ".txt")
            data = file.read()
            for c in range(len(self.characters)):
                counts[c] = counts[c] + data.count(self.characters[c])
        return(counts)
    
    def get_probabilities(self):
        N = sum(self.counts)
        probs = []
        for i in range(len(self.characters)):
            p = (self.counts[i] + 0.5) / (N + 27 * 0.5)
            probs.append(p)
        return(probs)
    
    def get_likelihood(self, filename):
        X = np.zeros(len(self.characters))
        file = open("homework4/data/languageID/" + filename + ".txt")
        data = file.read()
        for c in range(len(self.characters)):
            X[c] = data.count(self.characters[c])
        P = self.probabilities
        return(sum([b * math.log(a) for a, b in zip(P, X)]))

    def print_table(self, type, n_dec=4):
        if type == 1:
            values = self.counts
        elif type == 2:
            values = self.probabilities
        for i in range(len(self.characters)):
            print(self.characters[i], "&", round(values[i], n_dec), "\\\\")

# # print theta tables
# for l in range(len(L)):
#     print(L[l])
#     c = NaiveBayes(L[l], S, range(10))
#     c.print_table(2)

# # question 4
# e = NaiveBayes("e", S, [10])
# e.print_table(1)

# question 5
for l in range(len(L)):
    print(L[l])
    c = NaiveBayes(L[l], S, range(10))
    print(c.get_likelihood("e10"))

if __name__=='__main__':
    pass
