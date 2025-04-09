
import pandas as pd
import numpy as np
import networkx as nx
import community
import matplotlib.pyplot as plt
import umap
import os
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage

def get_geo_list(path:str):
    read =  pd.read_csv(path)
    read = read.loc[read['depository_source'] == 'GEO']
    read = read.loc[read['species'] == 'Arabidopsis thaliana']
    return list(read['depository_accession'])
def mapping(x):
    if type(x) is str:
        return x.upper()
    else:
        return x
    
def predicate(gene:str, chromosome:str)-> bool:
    return str('AT'+chromosome+'G') in gene

def get_first_indexs(df_index,chromo:list[str]):
    array = []
    for i in chromo:
        gene:str = next(filter(lambda x : predicate(x,str(i)), df_index))
        array.append(df_index.get_loc(gene))
    return array


def louvain_clustering(similarity_matrix, threshold=0.8):
    '''
    Cluster high-dimensional vectors using Louvain clustering.

    Parameters:
    vectors (np.ndarray): A 2D array of shape (n_vectors, n_features).
    threshold (float): Similarity threshold for creating edges in the graph.

    Returns:
    list: A list of clusters, where each cluster is a list of indices of similar vectors.
    '''


    # Step 2: Create a graph from the similarity matrix
    graph = nx.Graph()
    n = similarity_matrix.shape[0]

    # Add nodes
    for i in range(n):
        graph.add_node(i)

    # Add edges based on the similarity threshold
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                graph.add_edge(i, j, weight=similarity_matrix[i, j])

    # Step 3: Apply Louvain clustering
    partition = community.best_partition(graph, resolution=1.0)

    # Step 4: Organize the results into clusters
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    # Convert clusters to a list of lists
    clusters = list(clusters.values())

    return clusters

def get_Umap(matrix:np.array, study_map:list= None, name:str = '',save_loc: str = '', title:str = 'UMAP projection of the dataset'):
    # UMAP plotting
    reducer = umap.UMAP()
    scaled_data = StandardScaler().fit_transform(matrix)
    embedding = reducer.fit_transform(scaled_data)
    if study_map is None:
        plt.scatter(
        embedding[:, 0],
        embedding[:, 1]
        )
    else:
        colors = cm.rainbow(np.linspace(0, 1, max(study_map)+1))
        for num, emb in enumerate(embedding):
            plt.scatter(
                emb[0],
                emb[1]
                , color=colors[study_map[num]])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=24)
    plt.savefig(save_loc+'/umap'+name+'.svg')
    plt.close()

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def plot_sim_matrix(matrix:np.array,indices:list=None,chromosomes:list=None, name:str = '', save_loc: str = '', title:str= ''):
    # Plotting similarity matrix
    folder = 'Genes/'
    if indices is None:
        indices = [0]
        folder = 'Samples/'
    if chromosomes is None:
        chromosomes = ['']
    for i,c in enumerate(indices):
        print('plottin sim matrix', i)
        min = indices[i]
        try:
            max = indices[i+1]
        except:
            max = len(matrix)
        print('starting similarity')
        # Step 1: Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(matrix[min:max])
        print('starting plot')
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.savefig(save_loc+'/sim_matrix/'+folder+'sim_'+str(chromosomes[i])+'_matrix'+name+'.svg')
        plt.close()
        print('finished plot')

    plt.close()

    similarity_matrix = cosine_similarity(matrix)
    print('saving similarity_matrix')
    # np.save(save_loc+'sim_matrix.pny',similarity_matrix)
    print('starting final plot')
    print('done with final sim plot')

def apply_KNN_impute(df:pd.DataFrame,n_neighbors: int):
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Fit and transform the dataset
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df_imputed


def hierarchical_clustering(data_matrix:np.array):
    linkage_data = linkage(data_matrix, method='ward', metric='euclidean')
    return linkage_data

def hierarchical_clustering_plot(data_matrix:np.array, path:str, name:str):
    linkage_data = linkage(data_matrix, method='ward', metric='euclidean', optimal_ordering=True)
    dendrogram(linkage_data, no_labels= True)
    plt.savefig(path+'/cluster_ordered_'+name+'.svg')
    plt.close()


def box_plot(df: pd.DataFrame, cols_per_plot:int, out_path: str, group:int = 0):
    num_cols = len(df.columns)
    num_plots = math.ceil(num_cols / cols_per_plot)  # Calculate number of plots needed
    # Create directory for this group
    plot_path = os.path.join(out_path, f'boxplot_group_{group}')
    os.makedirs(plot_path, exist_ok=True)  # exist_ok prevents errors if dir exists

    for plot_num in range(num_plots):
        # Calculate start and end column indices for this plot
        start_idx = plot_num * cols_per_plot
        end_idx = min((plot_num + 1) * cols_per_plot, num_cols)
        
        
        # Create figure with appropriate size
        plt.figure(figsize=(20, 10))  # Adjust size as needed
        
        # Get columns for this plot and create boxplot
        current_cols = df.iloc[:, start_idx:end_idx]
        plt.boxplot(current_cols, labels=current_cols.columns)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add title and adjust layout
        plt.title(f'Boxplot Group {plot_num + 1} (Columns {start_idx + 1}-{end_idx})')
        plt.tight_layout()  # Prevents label cutoff
        
        # Save and close
        plt.savefig(os.path.join(plot_path, f'boxplot_group_{plot_num + 1}.png'))
        plt.close()