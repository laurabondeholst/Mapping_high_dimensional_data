"""
Takes the MNIST dataset as input (images and labels separated)
and creates a new dataset only with 0's and 1's
"""

import numpy as np

DATA_PATH = "data/raw/"
OUTPUT_PATH = "data/processed/mnist/"
X = np.loadtxt(DATA_PATH + "mnist2500_X.txt")
labels = np.loadtxt(DATA_PATH + "mnist2500_labels.txt")

X_new = []
labels_new = []

for i,label in enumerate(labels):
    if label < 5: 
        labels_new.append(label)
        X_new.append(X[i])
    if i%100 == 0: 
        print(f"{i} labels passed")

np.savetxt(OUTPUT_PATH + "mnist2500_X_01234.txt",X_new)
np.savetxt(OUTPUT_PATH +"mnist2500_labels_01234.txt",labels_new)