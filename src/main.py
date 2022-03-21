import numpy as np
import plotly.graph_objects as go
from t_sne_implementation import tsne
from validation import kmeans_clustering as kmeans


DATA_PATH = "data/processed/"
DATA_OUTPUT = DATA_PATH + "noisy_mnist/tsne_results/"
X = np.loadtxt(DATA_PATH + f"noisy_mnist/mnist2500_X_01_sigma10.txt")
labels = np.loadtxt(DATA_PATH + "mnist/mnist2500_labels_01.txt")

train_size = 0.5

percentiles = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
# percentiles = [1,2,3,4,5,6,7,8,9,10]
repetitions = np.array(100/np.array(percentiles))
repetitions = repetitions.astype(int)

accuracy = []
for perc_i, percentile in enumerate(percentiles):
    print(f"Running for percentile: {percentile}")
    # divide data into subsets equal to percentile * repetions
    np.random.seed(42) # seed is reset every time so we may run a single percentile alone and still get the same results
    
    perc_acc = []
    for rep in range(repetitions[perc_i]):
        if rep%10 == 0:
            print(f"Running repetition {rep} out of {repetitions[perc_i]}")
        arr_rand = np.random.rand(X.shape[0])
        split = arr_rand < np.percentile(arr_rand, percentile)
        
        X_split = X[split]
        labels_split = labels[split]

        Y = tsne.tsne(X = X_split)
        perc_acc.append(kmeans.run_kmeans(Y, labels_split, test_size=1-train_size))
    #   
    curr_acc = np.mean(perc_acc)
    accuracy.append(curr_acc) 
    print(f"Mean accuracy for percentile {percentile} is: {curr_acc}")
    # save error for percentile


fig = go.Figure()               
fig.add_trace(go.Scatter(x=percentiles, y=accuracy,
                    mode='lines'
                    ))
 
fig.show()