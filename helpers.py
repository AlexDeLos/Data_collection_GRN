
import pandas as pd
import numpy as np
import networkx as nx
import community
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

def get_geo_list(path:str):
    read =  pd.read_csv(path)
    read = read.loc[read['depository_source'] == "GEO"]
    read = read.loc[read['species'] == "Arabidopsis thaliana"]
    return list(read["depository_accession"])
def mapping(x):
    if type(x) is str:
        return x.upper()
    else:
        return x
    
def predicate(str:str, chromosome:str)-> bool:

    return "AT"+chromosome+"G" in str

def get_first_indexs(df_index,chromo:list[str]):
    array = []
    for i in chromo:
        gene:str = next(filter(lambda x : predicate(x,i), df_index))
        array.append(df_index.get_loc(gene))
    return array


def louvain_clustering(similarity_matrix, threshold=0.8):
    """
    Cluster high-dimensional vectors using Louvain clustering.

    Parameters:
    vectors (np.ndarray): A 2D array of shape (n_vectors, n_features).
    threshold (float): Similarity threshold for creating edges in the graph.

    Returns:
    list: A list of clusters, where each cluster is a list of indices of similar vectors.
    """


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

def get_Umap(matrix:np.array):
    # UMAP plotting
    reducer = umap.UMAP()
    scaled_data = StandardScaler().fit_transform(matrix)
    embedding = reducer.fit_transform(scaled_data)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1]
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the dataset', fontsize=24)
    plt.savefig("umap_no_nan.svg")

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
