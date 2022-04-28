import pandas as pd
import plotly.graph_objects as go
import numpy as np



UMAP_TSNE_FOLDER = "reports_from_tobias/reports/fashion_natural_umap_tsne/"
TSNE_FOLDER = "reports/Noiselevel_experiment_pca_tsne/Fashion/"
TRIMAP_FOLDER = "reports_from_pranjal/aml_results/mnist-strat/"

files = ["results_sigma0.csv", "results_sigma02.csv", "results_sigma05.csv", "results_sigma07.csv", "results_sigma1.csv"]
noiselevel = [0, 0.2, 0.5, 0.7, 1]

# files = ["results_sigma0.csv", "results_sigma02.csv", "results_sigma05.csv","results_sigma1.csv"]
# noiselevel = [0, 0.2, 0.5, 1]

for i,file in enumerate(files): 
    # umap_tsne_df = pd.read_csv(UMAP_TSNE_FOLDER + file)
    trimap_df = pd.read_csv(TRIMAP_FOLDER + file)


        
    fig = go.Figure(layout_xaxis_range=[0,np.max(trimap_df.data_points_number)],layout_yaxis_range=[0,1])
    # fig.add_trace(go.Scatter(x=umap_tsne_df.data_points_number.values, y=umap_tsne_df.correct_predicted_percent_pca.values, name="PCA", mode='lines'))
    fig.add_trace(go.Scatter(x=trimap_df.data_points_number.values, y=trimap_df.correct_predicted_percent_pca.values, name="PCA", mode='lines', fillcolor='green'))
    
    fig.add_trace(go.Scatter(x=trimap_df.data_points_number.values, y=trimap_df.correct_predicted_percent_trimap.values, name="TRIMAP", mode='lines', fillcolor='blue'))
    fig.add_trace(go.Scatter(x=trimap_df.data_points_number.values, y=trimap_df.correct_predicted_percent_tsne.values, name="TSNE", mode='lines', fillcolor='red'))
    fig.add_trace(go.Scatter(x=trimap_df.data_points_number.values, y=trimap_df.correct_predicted_percent_umap.values, name="UMAP", mode='lines', fillcolor='purple'))

    fig.update_layout(title="MNIST stratified distribution", legend_title_text = "Noise level: {}".format(noiselevel[i]))
    fig.update_xaxes(title_text="Datapoints")
    fig.update_yaxes(title_text="Accuracy [%]")
    fig.show()

    