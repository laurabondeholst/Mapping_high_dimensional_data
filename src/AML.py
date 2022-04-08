# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits

import umap  # "pip install umap-learn --ignore-installed" does the trick for Laura
# import trimap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import warnings

from t_sne_implementation import tsne
from validation import kmeans_clustering as kmeans

import plotly.graph_objects as go

plt.style.use('fivethirtyeight') # For better style
warnings.filterwarnings("ignore")

trimap_enable = False
tsne_enable = True
umap_enable = True

# DATA_PATH = "data/processed/"
# DATA_OUTPUT = DATA_PATH + "noisy_mnist/tsne_results/"
# X = np.loadtxt(DATA_PATH + f"noisy_mnist/mnist2500_X_01_sigma10.txt")
# labels = np.loadtxt(DATA_PATH + "mnist/mnist2500_labels_01.txt")



def addnoise( mu, sigma, X):
  
  noise = np.random.normal(mu, sigma, X.shape) 
  noisy_X = X + noise

  return normalise(noisy_X)

def normalise(X):
  mini = np.min(X)
  maxi = np.max(X)
  return (X-mini)/maxi

def distributeData(X, y, min_class_size, classes = [0,1]):
  """
  Will stratify the data unevenly, so that the first class is min_size large

  min_class_size should be float between 0 and 0.5

  Returns: X, y
  """
  index0 = np.where(y==classes[0])
  index1 = np.where(y==classes[1])

  temp_x0 = X[index0[0]]
  temp_y0 = y[index0[0]]
  temp_x1 = X[index1[0]]
  temp_y1 = y[index1[0]]

  print(f"Previous perc of class 0 between classes: {len(temp_x0)/(len(temp_x0)+len(temp_x1))} ")
  print(f"Previous perc of class 1 between classes: {len(temp_x1)/(len(temp_x0)+len(temp_x1))} ")
  arr_rand = np.random.rand(len(temp_x0))
  split = arr_rand < np.percentile(arr_rand,100*(min_class_size/(1-min_class_size)))

  temp_x0 = temp_x0[split]
  temp_y0 = temp_y0[split]


  print(f"New perc of class 0 between classes: {len(temp_x0)/(len(temp_x0)+len(temp_x1))}")
  print(f"New perc of class 1 between classes: {len(temp_x1)/(len(temp_x0)+len(temp_x1))}")

  new_X = np.concatenate((temp_x0,temp_x1))
  new_y = np.concatenate((temp_y0,temp_y1))
  return new_X, new_y



## INITIALISING VALUES
n_classes=2
digits = load_digits(n_class=n_classes)
dataset_length= len(digits.data);

digits_0 = load_digits(n_class=1)
dataset_length_0= len(digits_0.data);

digits.data, digits.target = distributeData(digits.data, digits.target, min_class_size = 0.25)
dataset_length = len(digits.target)


X_norm  = []
for digit in digits.data:
  X_norm.append(normalise(digit))


crossvaltimes= 5;
noise_range=[0,10]
testing_range=75
train_size = 0.6

# progress_bar = tqdm(range(noise_range*(testing_range-4)))

correct_count_list_pca = []
correct_count_list_trimap=[];
correct_count_list_tsne=[];
correct_count_list_umap=[];


ns_i_list = []

datapoint_range = []
for i in range(4,int(dataset_length)): # from 4 since some models requires at least 3 datapoints
  if i < dataset_length*0.1:
    datapoint_range.append(i)
  elif i < dataset_length*0.3 and i%10 == 0: 
    datapoint_range.append(i)
  elif i%50 == 0: 
    datapoint_range.append(i)

repetition_range = []
for i in datapoint_range:
  perc = i/dataset_length*100
  if perc < 1:
    perc = 1

  repetition_range.append(int(100/perc))


## MAIN LOOP
for ns in noise_range: 
  noisy_X= addnoise(0,ns,np.array(X_norm))

  progress_bar = tqdm(np.array(datapoint_range)*np.array(repetition_range))

  correct_count_list_pca = []
  correct_count_list_trimap=[]
  correct_count_list_tsne=[]
  correct_count_list_umap=[]


  ns_i_list = []
  

  for i_enum, i in enumerate(datapoint_range):
    correct_count_pca=[]
    correct_count_trimap=0
    correct_count_tsne=[]
    correct_count_umap=[]
    np.random.seed(42)
    for jr in range(repetition_range[i_enum]):
      # split into train test sets
      X, _, y, _ = train_test_split(noisy_X, digits.target, train_size=(i/dataset_length), stratify=digits.target ) 

      y_pred= pca = PCA(n_components=3).fit_transform(X)
      correct_count_pca.append(kmeans.run_kmeans(y_pred, y, test_size=0.5))

      if trimap_enable: 
        if(i<=4):
          n_in= i-2
        else:
          n_in=3
        embedding_trimap = trimap.TRIMAP(n_inliers=n_in, n_outliers=3, n_random=3).fit_transform(X)
        kmeans_trimap = KMeans(n_clusters=2, random_state=0).fit(embedding_trimap);
        predicted_labels_trimap= kmeans_trimap.labels_;
        y_find_trimap, y_analyze_trimap, pred_find_trimap, pred_analyze_trimap = train_test_split(y, predicted_labels_trimap, train_size=0.5,stratify=y)
      
        count1=0; 

        for cj in range(0,len(y_find_trimap)):
          if(y_find_trimap[cj]== pred_find_trimap[cj]):
            count1= count1+1;


        if (count1 >= (len(y_find_trimap)/2)):
          for cj in range (0, len(y_analyze_trimap)):
            if(y_analyze_trimap[cj]== pred_analyze_trimap[cj]):
              correct_count_trimap= correct_count_trimap+1;
        else:
          for cj in range (0, len(y_analyze_trimap)):
            if(y_analyze_trimap[cj]!= pred_analyze_trimap[cj]):
              correct_count_trimap= correct_count_trimap+1;      

      if tsne_enable: 
        y_pred = TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
        correct_count_tsne.append(kmeans.run_kmeans(y_pred, y, test_size=0.5))

      if umap_enable: 
        y_pred = umap.UMAP().fit_transform(X)
        correct_count_umap.append(kmeans.run_kmeans(y_pred, y, test_size=0.5))


    ns_i_list.append([ns, i])
    correct_count_list_pca.append(np.mean(correct_count_pca))

    if trimap_enable: 
      correct_count_list_trimap.append(((correct_count_trimap*200)/(i*crossvaltimes)))
    if tsne_enable: 
      correct_count_list_tsne.append(np.mean(correct_count_tsne))
    if umap_enable: 
      correct_count_list_umap.append(np.mean(correct_count_umap))



    progress_bar.update(1)
    
  # Create the pandas DataFrame
  df = pd.DataFrame(ns_i_list, columns=['Noise_sigma', 'data_points_number'])

  df['correct_predicted_percent_pca'] = correct_count_list_pca


  if trimap_enable: 
    df['correct_predicted_percent_trimap'] = correct_count_list_trimap
  if tsne_enable: 
    df['correct_predicted_percent_tsne'] = correct_count_list_tsne
  if umap_enable: 
    df['correct_predicted_percent_umap'] = correct_count_list_umap

  df.to_csv(f'results_sigma{ns}.csv')

  fig = go.Figure()
  
  fig.add_trace(go.Scatter(x=df.data_points_number.values, y=df.correct_predicted_percent_pca.values, name="PCA", mode='lines'))
  
  if trimap_enable:
    fig.add_trace(go.Scatter(x=df.data_points_number.values, y=df.correct_predicted_percent_trimap.values, name="TRIMAP", mode='lines'))
  if tsne_enable:
    fig.add_trace(go.Scatter(x=df.data_points_number.values, y=df.correct_predicted_percent_tsne.values, name="TSNE", mode='lines'))
  if umap_enable:
    fig.add_trace(go.Scatter(x=df.data_points_number.values, y=df.correct_predicted_percent_umap.values, name="UMAP", mode='lines'))

  fig.update_layout(legend_title_text = f"Noise level: {ns}")
  fig.update_xaxes(title_text="Datapoints")
  fig.update_yaxes(title_text="Accuracy [%]")
  fig.show()
  
