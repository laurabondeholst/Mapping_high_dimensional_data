"""
This script takes the coordinates of the mapping method (the output from t-SNE in my case)
as well as the true labels of the correndsponding data and produces a plot of the error rate

TODO: modify the ranges to fit your data
TODO: modify the filepath and names of the output from the mapping method and true labels

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import plotly.graph_objects as go

DATA_PATH = "data/processed/noisy_mnist/tsne_results/"


def run_kmeans(Y, labels, n_clusters = 2, test_size = 0.5):
    Y_train, Y_test, labels_train, labels_test = train_test_split(Y, labels, test_size=test_size, random_state=42)

    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        #  n_init=10,
        max_iter=300,
        random_state=42
    )

    kmeans.fit(Y_train)

    y_pred = kmeans.labels_

    # if n_clusters != 2: ## not working: several clusters get assigned same id
    #     y_pred_new = y_pred
    #     for cluster in range(n_clusters):
    #         idx = np.where(y_pred == cluster)
    #         subset_labels = labels_train[idx]
    #         subset_labels = np.sort(subset_labels)
    #         med = np.median(subset_labels)

    #         y_pred_new[idx] = med

    #     y_pred = y_pred_new
        
    mistakes_true_labels = 0
    mistakes_inverted_labels = 0
    for i in range(len(y_pred)):
        if y_pred[i] != labels_train[i]:
            mistakes_true_labels += 1
        else: 
            mistakes_inverted_labels += 1
    
    # we assume that correct labels have smallest error
    if mistakes_true_labels < mistakes_inverted_labels: 
        label_map = {0: 0, 1: 1}
    else:
        label_map = {0: 1, 1: 0}

    y_pred_test = kmeans.predict(Y_test)

    mistakes = 0
    for i in range(len(y_pred_test)):
        assumed_label = label_map[y_pred_test[i]]
        if assumed_label != labels_test[i]:
            mistakes += 1
        
    # mistakes = mistakes_e1 if (mistakes_e1 < mistakes_e2) else mistakes_e2
    return 1 - mistakes/len(y_pred_test)



if __name__ == "__main__":
    error = []
    perc = []

    i = 5
    sigma = 50

            
    for percentile in range(1,10,1):
        # Y = np.loadtxt(DATA_PATH + f"TSNE_output_{percentile}_sigma{sigma}.txt") 
        # labels = np.loadtxt(DATA_PATH + f"true_labels_{percentile}_sigma{sigma}.txt")

        # run_kmeans(Y, labels, percentile, error[i], perc[i], 2)

        Y = np.loadtxt(DATA_PATH + f"TSNE_output_{percentile}_sigma{sigma}.txt") 
        labels = np.loadtxt(DATA_PATH + f"true_labels_{percentile}_sigma{sigma}.txt")

        error.append(run_kmeans(Y, labels, 2))
        perc.append(percentile)

    for percentile in range(10,110,10):
    #     # Y = np.loadtxt(DATA_PATH +f"TSNE_output_{percentile}_sigma{sigma}.txt") 
    #     # labels = np.loadtxt(DATA_PATH +f"true_labels_{percentile}_sigma{sigma}.txt")

    #     # run_kmeans(Y, labels, percentile, error[i], perc[i], 2)

        Y = np.loadtxt(DATA_PATH + f"TSNE_output_{percentile}_sigma{sigma}.txt") 
        labels = np.loadtxt(DATA_PATH + f"true_labels_{percentile}_sigma{sigma}.txt")

        error.append(run_kmeans(Y, labels, 2))
        perc.append(percentile)

    # for j in range(len(error)): 
    #     curr_err = error[j]
    #     if curr_err < 0.4:
    #         error[j] = 1-curr_err

    fig = go.Figure()

    # for i in range(len(perc)):
    #     fig.add_trace(go.Scatter(x=perc[i], y=error[i],
    #                     mode='lines',
    #                     name=i+1
    #                     ))

    fig.add_trace(go.Scatter(x=perc, y=error,
                        mode='lines'
                        ))
    
    fig.show()

