import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np  
from helpers import plot_sim_matrix,get_Umap, apply_KNN_impute,hierarchical_clustering_plot,box_plot

def hierarchical_clustering_with_colinearity(df, threshold=0.9):
    """
    Perform hierarchical clustering ensuring clusters have 90% colinearity.
    
    Parameters:
    - df: pandas DataFrame (samples as rows, features as columns)
    - threshold: colinearity threshold (default 0.9 for 90%)
    
    Returns:
    - Cluster labels for each sample
    """
    
    # Calculate pairwise correlation matrix
    corr_matrix = df.corr().values
    
    # Convert correlation to distance (1 - absolute correlation)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)  # set diagonal to 0
    
    # Perform hierarchical clustering
    Z = linkage(squareform(distance_matrix), method='complete')
    
    # Determine clusters based on the colinearity threshold
    clusters = fcluster(Z, 1 - threshold, criterion='distance')
    
    return clusters

path = '/tudelft.net/staff-umbrella/AT GE Datasets/df_local/df/'
# path = 'df/'
names = ['corrected','robust','standardized']
out_path = '/tudelft.net/staff-umbrella/AT GE Datasets/clustered_figures'
# out_path = 'clustered_figures'
th_list = [0.9,0.8,0.7,0.6,0.5]
for name in names:
    for th in th_list:
        try:
            print('reading')
            # raise FileNotFoundError
            cluster_means = pd.read_csv(path+'averaged_clustered_'+str(th)+'_'+name+'.csv', index_col=0)
        except FileNotFoundError as e:
            data = pd.read_csv(path+name+'.csv', index_col=0)
            clusters = hierarchical_clustering_with_colinearity(data.T,threshold = th)
            data['cluster'] = clusters
            # data.to_csv(path+'clustered_'+name)
            # df_clust = pd.read_csv(path+'clustered_'+name+'_'+str(th), index_col=0)
            # Group by cluster and take the mean of each cluster
            cluster_means = data.groupby('cluster').mean()
            print('For theshold ',th, 'we found ', len(cluster_means),' clusters')
            # Save the averaged clusters
            cluster_means.to_csv(path+'averaged_clustered_'+str(th)+'_'+name+'.csv')


        def get_study(sample: str):
            return int(sample.split('_')[-1])
        study_map = list(map(get_study,cluster_means.columns))

        print("Starting plotting")
        # plot_sim_matrix(cluster_means,name='clustered', save_loc=out_path, title= 'Data full sim mat (clustered)')
        get_Umap(cluster_means.to_numpy(),name='_genes_'+name+'_'+str(th),save_loc=out_path, title='Gene expressions (clustered)')
        get_Umap(cluster_means.to_numpy().T,name='_samples_'+name+'_'+str(th),study_map=study_map,save_loc=out_path, title='Samples coloured by study (clustered)')
        plot_sim_matrix(cluster_means.to_numpy().T,name='_'+name+'_'+str(th),save_loc=out_path,title='Samples full sim mat (clustered)')
        print("Done plotting")

