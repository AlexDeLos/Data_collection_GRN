import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import numpy as np  

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
name= 'robust.csv'
data = pd.read_csv(path+name, index_col=0)
clusters = hierarchical_clustering_with_colinearity(data.T)
data['cluster'] = clusters
data.to_csv(path+'clustered_robust.csv')


# df_clust = pd.read_csv('df/clustered.csv', index_col=0)

# x= 0