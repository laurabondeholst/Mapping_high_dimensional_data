"""
This script takes the coordinates of the mapping method (the output from t-SNE in my case)
as well as the true labels of the correndsponding data and produces a plot of the error rate

TODO: modify the ranges to fit your data
TODO: modify the filepath and names of the output from the mapping method and true labels

"""


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def run_kmeans(percentile, error, perc):
    # TODO: modify names
    Y = np.loadtxt(f"TSNE_output_{percentile}.txt") 
    labels = np.loadtxt(f"true_labels_{percentile}.txt")

    kmeans = KMeans(
        init="random",
        n_clusters=2,
        #  n_init=10,
        max_iter=300,
        random_state=42
    )

    kmeans.fit(Y)

    y_pred = kmeans.labels_


    mistakes = 0
    for i in range(len(y_pred)):
        if y_pred[i] != labels[i]:
            mistakes += 1

    error.append(mistakes/len(y_pred))
    perc.append(percentile)

error = []
perc = []

# TODO: modify ranges
for percentile in range(1,10,1):
   run_kmeans(percentile, error, perc)

for percentile in range(10,110,10):
   run_kmeans(percentile, error, perc)

for i in range(len(error)): 
    curr_err = error[i]
    if curr_err < 0.2:
        error[i] = 1-curr_err

plt.plot(perc,error)
plt.show()

