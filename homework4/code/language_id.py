import numpy as np
import pandas as pd
import math

# character types
S = [chr(i) for i in range(ord('a'), ord('z') + 1)]
S.append(" ")

# labels
L = ['e', 'j', 's']

class NaiveBayes:

    def __init__(self, label, characters, files, alpha = 0.5):
        self.label = label
        self.characters = characters
        self.files = files
        self.alpha = alpha
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
            p = (self.counts[i] + self.alpha) / (N + 27 * self.alpha)
            probs.append(p)
        return(probs)
    
    def get_log_lik(self, filename):
        X = np.zeros(len(self.characters))
        file = open("homework4/data/languageID/" + filename + ".txt")
        data = file.read()
        for c in range(len(self.characters)):
            X[c] = data.count(self.characters[c])
        P = self.probabilities
        return(sum([x * math.log(p) for p, x in zip(P, X)]))

    def print_table(self, type, n_dec=4):
        if type == 1:
            values = self.counts
        elif type == 2:
            values = self.probabilities
        for i in range(len(self.characters)):
            print(self.characters[i], "&", round(values[i], n_dec), "\\\\")

if __name__=='__main__':
    # # print theta tables
    # for l in range(len(L)):
    #     print(L[l])
    #     c = NaiveBayes(L[l], S, range(10))
    #     c.print_table(2)

    # # question 4
    # e = NaiveBayes("e", S, [10])
    # e.print_table(1)

    # # question 5
    # post = []
    # for l in range(len(L)):
    #     print(L[l])
    #     c = NaiveBayes(L[l], S, range(10))
    #     post.append(c.get_log_lik("e10"))
    #     print(post[l])

    # question 7
    models = []
    for m in range(len(L)):
        models.append(NaiveBayes(L[m], S, range(10)))
    
    print("predicted", "actual")
    for l in range(len(L)):
        for f in range(10, 20):
            name = L[l] + str(f)
            log_lik = []
            for m in range(len(models)):
                log_lik.append(models[m].get_log_lik(name))
            max_index = np.argmax(log_lik)
            print(L[max_index], L[l])

    pass
