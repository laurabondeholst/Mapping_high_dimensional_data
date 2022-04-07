# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import time
import umap 
import trimap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import warnings
import pandas as pd

plt.style.use('fivethirtyeight') # For better style
warnings.filterwarnings("ignore")





def addnoise( mu, sigma, X):
  
  noise = np.random.normal(mu, sigma, X.shape) 
  noisy_X = X + noise

  return (noisy_X)




df_fashion = pd.read_csv("fashion-mnist_train.csv")

rslt_df = df_fashion[(df_fashion['label'] == 0) | 
          (df_fashion['label'] == 8)] 
    

fashion_y = rslt_df['label'].to_numpy()
fashion_X = rslt_df.drop(columns=[ 'label']).to_numpy()



dataset_length= len(fashion_y);



crossvaltimes= 10;
noise_range=100;
testing_range=300;

progress_bar = tqdm(range(int((noise_range+1)*(testing_range-4)/4)))

correct_count_list_trimap=[];
correct_count_list_tsne=[];
correct_count_list_umap=[];
correct_count_list_pca=[];

for ns in range(0,(noise_range+1),5):
  noisy_X= addnoise(0,ns,fashion_X) 
  norm_noisy_X= StandardScaler().fit_transform(noisy_X)
  for i in range (4, testing_range,4):
    total_elements_evaluated_trimap=0;
    total_elements_evaluated_umap=0;
    total_elements_evaluated_tsne=0;
    total_elements_evaluated_pca= 0;
    correct_count_trimap=0;
    correct_count_tsne=0;
    correct_count_umap=0;
    correct_count_pca= 0;
    if(i<=10):
        crossvaltimes=20;
    elif(i<=20):
        crossvaltimes=10;
    elif(i<=50):
        crossvaltimes=5;
    elif(i<=100):
        crossvaltimes=3;
    else:
        crossvaltimes=2;
    for jr in range (0,crossvaltimes):
      # split into train test sets
      X, X_test, y, y_test = train_test_split(norm_noisy_X, fashion_y, train_size=(i/dataset_length), stratify= fashion_y)

      if(i<=4):
        n_in= i-2;
      else:
        n_in=3;
      embedding_trimap = trimap.TRIMAP(n_inliers=n_in, n_outliers=3, n_random=3).fit_transform(X);
      kmeans_trimap = KMeans(n_clusters=2, random_state=0).fit(embedding_trimap);
      predicted_labels_trimap= kmeans_trimap.labels_;

      embedding_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(X);
      kmeans_tsne = KMeans(n_clusters=2, random_state=0).fit(embedding_tsne);
      predicted_labels_tsne= kmeans_tsne.labels_;

      embedding_umap = umap.UMAP().fit_transform(X);
      kmeans_umap = KMeans(n_clusters=2, random_state=0).fit(embedding_umap);
      predicted_labels_umap= kmeans_umap.labels_;
      
      embedding_pca= pca = PCA(n_components=3).fit_transform(X)
      kmeans_pca = KMeans(n_clusters=2, random_state=0).fit(embedding_pca);
      predicted_labels_pca= kmeans_pca.labels_;
      
      
      
      y_find_trimap, y_analyze_trimap, pred_find_trimap, pred_analyze_trimap = train_test_split(y, predicted_labels_trimap, train_size=0.5,stratify=y);
      y_find_tsne, y_analyze_tsne, pred_find_tsne, pred_analyze_tsne = train_test_split(y, predicted_labels_tsne, train_size=0.5,stratify=y);
      y_find_umap, y_analyze_umap, pred_find_umap, pred_analyze_umap = train_test_split(y, predicted_labels_umap, train_size=0.5,stratify=y);
      y_find_pca, y_analyze_pca, pred_find_pca, pred_analyze_pca = train_test_split(y, predicted_labels_pca, train_size=0.5,stratify=y);
       

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

      total_elements_evaluated_trimap= total_elements_evaluated_trimap+ len(y_analyze_trimap)

      count2=0; 
      for j in range(len(y_find_tsne)):
        if(y_find_tsne[j]== pred_find_tsne[j]):
          count2= count2+1;

      if (count2 >= (len(y_find_tsne)/2)):
        for j in range (0, len(y_analyze_tsne)):
          if(y_analyze_tsne[j]== pred_analyze_tsne[j]):
            correct_count_tsne= correct_count_tsne+1;
      else:
        for j in range (0, len(y_analyze_tsne)):
          if(y_analyze_tsne[j]!= pred_analyze_tsne[j]):
            correct_count_tsne= correct_count_tsne+1;

      total_elements_evaluated_tsne= total_elements_evaluated_tsne+ len(y_analyze_tsne)

      count3=0; 
      for j in range(len(y_find_umap)):
        if(y_find_umap[j]== pred_find_umap[j]):
          count3= count3+1;

      if (count3 >= (len(y_find_umap)/2)):
        for j in range (0, len(y_analyze_umap)):
          if(y_analyze_umap[j]== pred_analyze_umap[j]):
            correct_count_umap= correct_count_umap+1;
      else:
        for j in range (0, len(y_analyze_umap)):
          if(y_analyze_umap[j]!= pred_analyze_umap[j]):
            correct_count_umap= correct_count_umap+1;
      total_elements_evaluated_umap= total_elements_evaluated_umap+ len(y_analyze_umap)
    
    
    count4=0; 
    for j in range(len(y_find_pca)):
      if(y_find_pca[j]== pred_find_pca[j]):
        count4= count4+1;

    if (count4 >= (len(y_find_pca)/2)):
      for j in range (0, len(y_analyze_pca)):
        if(y_analyze_pca[j]== pred_analyze_pca[j]):
          correct_count_pca= correct_count_pca+1;
    else:
      for j in range (0, len(y_analyze_pca)):
        if(y_analyze_pca[j]!= pred_analyze_pca[j]):
          correct_count_pca= correct_count_pca+1;
    total_elements_evaluated_pca= total_elements_evaluated_pca+ len(y_analyze_pca)
  
            
    correct_count_list_trimap.append([ns,i,((correct_count_trimap*100)/(total_elements_evaluated_trimap))]);
    correct_count_list_tsne.append(((correct_count_tsne*100)/(total_elements_evaluated_tsne)));
    correct_count_list_umap.append(((correct_count_umap*100)/(total_elements_evaluated_umap)));
    correct_count_list_pca.append(((correct_count_pca*100)/(total_elements_evaluated_pca)));



    progress_bar.update(1)
    
  # Create the pandas DataFrame
  df = pd.DataFrame(correct_count_list_trimap, columns = ['Noise_sigma', 'data_points_number', 'correct_predicted_percent_trimap']);
  df['correct_predicted_percent_tsne'] = correct_count_list_tsne;
  df['correct_predicted_percent_umap'] = correct_count_list_umap;
  df['correct_predicted_percent_pca'] = correct_count_list_pca;




overlapping = 0.2

for ns in range(0,(noise_range+1)):
  df_focus= df.loc[df['Noise_sigma'] == ns]
  fig, ax = plt.subplots(figsize=(15,6))
  bp= ax.plot(df_focus.data_points_number.values, df_focus.correct_predicted_percent_trimap.values, "-y", label="TRIMAP", alpha=overlapping, linestyle= '-');
  bp= ax.plot(df_focus.data_points_number.values, df_focus.correct_predicted_percent_tsne.values, "-r", label="TSNE", alpha=overlapping , linestyle= '-.');
  bp= ax.plot(df_focus.data_points_number.values, df_focus.correct_predicted_percent_umap.values, "-b", label="UMAP", alpha=overlapping, linestyle= ':');
  bp= ax.plot(df_focus.data_points_number.values, df_focus.correct_predicted_percent_pca.values, "-g", label="PCA", alpha=overlapping, linestyle= '--');
  #setting x and y axis label
  ax.set_xlabel('Number of data points') ;
  ax.set_ylabel('Percentage of data points correctly classified') ;
  ax.set_title('Noise sigma = ' + str(ns) )# + "   Cross Validation times = " + str(crossvaltimes))
  ax.set_xlim(3, testing_range)
  ax.set_ylim(0, 105);
  ax.legend(loc="upper right")
  
  