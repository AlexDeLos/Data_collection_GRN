import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import get_first_indexs,louvain_clustering

from sklearn.metrics.pairwise import cosine_similarity

plot_nan = False

# Get CSV files list from a folder
path = 'df_no_nan'
# path = 'df'
csv_files = glob.glob(path + "/*.csv")

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file, index_col=0) for file in csv_files)

# Concatenate all DataFrames
big_df = pd.concat(df_list, axis=1)
big_df = big_df.dropna(axis=1, how='all')


#! Some duplicates have been created
big_df = big_df.loc[:,~big_df.columns.duplicated()].copy()
big_df.sort_index(inplace=True)


matrix = big_df.to_numpy()
matrix_nan = big_df.isna().to_numpy()

chromosomes = ["1",'2','3','4','5','C','M']
indices:list[int] = get_first_indexs(big_df.index,chromosomes)

if plot_nan:
    row_nan_count = big_df.isna().sum(axis=1)
    filter_row = big_df.isna().sum(axis=1)>1000
    col_nan_count = big_df.isna().sum(axis=0)

    matrix = big_df.to_numpy()
    

    plt.imshow(matrix_nan, cmap='hot', interpolation='nearest')
    plt.savefig('figures/matrix.svg')
    plt.close()


    
    for i,c in enumerate(indices):
        min = indices[i]
        try:
            max = indices[i+1]
        except:
            max = len(matrix)
        plt.bar(range(len(row_nan_count.index[min:max])),row_nan_count.values[min:max])
        plt.ylim(0,1850)
        plt.savefig('figures/row/row_dis'+chromosomes[i]+'.svg')
        plt.close()

        plt.bar(range(len(col_nan_count.index[min:max])),col_nan_count.values[min:max])
        plt.savefig('figures/col/col_dis'+chromosomes[i]+'.svg')
        plt.close()

np.nan_to_num(matrix,copy=False)
for i,c in enumerate(indices):
    min = indices[i]
    try:
        max = indices[i+1]
    except:
        max = len(matrix)
    print("starting similarity")
    # Step 1: Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(matrix[min:max])
    print("starting plot")
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('figures/sim_matrix/2_avg_sim_'+str(chromosomes[i])+'_matrix.svg')
    plt.close()
    print("finished plot")

    louvain_clustering(similarity_matrix)


print("Done")




# def get_Umap(matrix:np.array):
#     # UMAP plotting
#     reducer = umap.UMAP()
#     scaled_penguin_data = StandardScaler().fit_transform(matrix)
#     embedding = reducer.fit_transform(scaled_penguin_data)

#     plt.scatter(
#         embedding[:, 0],
#         embedding[:, 1]
#         )
#     plt.gca().set_aspect('equal', 'datalim')
#     plt.title('UMAP projection of the dataset', fontsize=24)
#     plt.savefig("umap.svg")

# get pair wise similar


