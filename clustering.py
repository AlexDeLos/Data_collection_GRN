import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as spc
import numpy as np
from helpers import plot_sim_matrix,get_Umap,plot_heat_map

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
    
    pdist = spc.distance.pdist(corr_matrix)

    # Perform hierarchical clustering
    Z = linkage(pdist, method='complete')
    
    # Determine clusters based on the colinearity threshold
    clusters = fcluster(Z, (1-threshold) * pdist.max(), criterion='distance')
    
    return clusters


def get_study(sample: str):
    return int(sample.split('_')[-1])


path = '/tudelft.net/staff-umbrella/AT GE Datasets/processed_final/'
# path = 'df_final/'
out_path = '/tudelft.net/staff-umbrella/AT GE Datasets/clustered_figures_final'
# out_path = 'clustered_figures_1.0'

print('starting')
names = ['corrected','robust','standardized']
chromosomes = ['a']
th_list = [0.85,0.8]

for name in names:
    data = pd.read_csv(path+name+'.csv', index_col=0)
    print('Starting:', name)
    plot_heat_map(data,out_path,name)
    for th in th_list:
        print('Starting:', th)
        try:
            # raise FileNotFoundError
            cluster_means = pd.read_csv(path+'averaged_clustered_'+str(th)+'_'+name+'.csv', index_col=0)
            data = pd.read_csv(path+'clustered_'+str(th)+'_'+name+'.csv', index_col=0)
        except FileNotFoundError as e:
            print("Creating the data for:",name,th)
            data = pd.read_csv(path+name+'.csv', index_col=0)
            clusters = hierarchical_clustering_with_colinearity(data.T,threshold = th)
            data['cluster'] = clusters
            data.to_csv(path+'clustered_'+str(th)+'_'+name+'.csv')
            # Group by cluster and take the mean of each cluster
            cluster_means = data.groupby('cluster').mean()
            # Save the averaged clusters
            cluster_means.to_csv(path+'averaged_clustered_'+str(th)+'_'+name+'.csv')
        df_clust = data
        print('For theshold ',th, 'we found ', len(cluster_means),' clusters')

        chunks = len(chromosomes)
        indx = []
        chunck_size = len(cluster_means)//chunks
        for chunk in range(chunks):
            indx.append(chunk*chunck_size)
        # indx.append(len(cluster_means))


        chunks_ = len(chromosomes)
        indx_ = []
        chunck_size_ = len(df_clust)//chunks_
        for chunk_ in range(chunks_):
            indx_.append(chunk_*chunck_size_)
        # indx_.append(len(df_clust))


        cluster_means.sort_index(inplace=True)
        df_clust.sort_values('cluster', inplace=True)
        study_map = list(map(get_study,cluster_means.columns))
        

        print("Starting plotting")
        # plot sim matrix 
        plot_sim_matrix(cluster_means,indx,chromosomes,name='_Gene_'+name+'_'+str(th),save_loc=out_path,title='Gen Sim (Not clustered)')
        plot_sim_matrix(df_clust,indx_,chromosomes,name='_Cluster_'+name+'_'+str(th),save_loc=out_path,title='Cluster Sim (Clustered)')

        get_Umap(cluster_means.to_numpy(),name='_genes_'+name+'_'+str(th),save_loc=out_path, title='Gene expressions (clustered)')
        get_Umap(cluster_means.to_numpy().T,name='_samples_'+name+'_'+str(th),study_map=study_map,save_loc=out_path, title='Samples coloured by study (clustered)')
        plot_sim_matrix(cluster_means.to_numpy().T,name='_'+name+'_'+str(th),save_loc=out_path,title='Samples full sim mat (clustered)')
        print("Done plotting")

