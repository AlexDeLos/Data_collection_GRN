import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import get_first_indexs,plot_sim_matrix,get_Umap, normalize_2d,apply_KNN_impute



plot_nan = True

# Get CSV files list from a folder
path = '/tudelft.net/staff-umbrella/AT GE Datasets/df_nan'
out_path = '/tudelft.net/staff-umbrella/AT GE Datasets/figures'

# path = 'df'
# out_path = 'figures'
try:
    filtered_df = pd.read_csv(path+'/filter.csv', index_col=0)
    print('succesfully loaded data')
except:
    csv_files = glob.glob(path + '/*.csv')

    # Read each CSV file into DataFrame
    # This creates a list of dataframes
    df_list = (pd.read_csv(file, index_col=0) for file in csv_files)

    # Concatenate all DataFrames
    big_df = pd.concat(df_list, axis=1)
    big_df = big_df.dropna(axis=1, how='all')

    #! Some duplicates have been created
    big_df = big_df.loc[:,~big_df.columns.duplicated()].copy()
    big_df.sort_index(inplace=True)

    big_df.to_csv(path+'/complete.csv')

    # filter the data on 20%
    nan_percentage_genes = big_df.isna().mean(axis=1) * 100

    # Filter rows where NaN percentage <= 20%
    filtered_genes = nan_percentage_genes[nan_percentage_genes <= 20].index

    # Keep only the filtered rows
    filtered_df = big_df.loc[filtered_genes]


    # Calculate the percentage of NaN values in each column
    nan_percentage_samples = filtered_df.isna().mean() * 100

    # Filter columns where NaN percentage <= 20%
    filtered_columns = nan_percentage_samples[nan_percentage_samples <= 20].index

    # Keep only the filtered columns
    filtered_df = filtered_df[filtered_columns]

    filtered_df.to_csv(path+'/filter.csv')

print('data loaded')
big_df = filtered_df
print(big_df.head)

matrix = big_df.to_numpy()
matrix_nan = big_df.isna().to_numpy()

chromosomes = ['1','2','3','4','5']
indices:list[int] = get_first_indexs(big_df.index,chromosomes)

if plot_nan:
    row_nan_count = big_df.isna().sum(axis=1)
    filter_row = big_df.isna().sum(axis=1)>1000
    col_nan_count = big_df.isna().sum(axis=0)
    
    plt.imshow(matrix_nan, cmap='hot', interpolation='nearest')
    plt.savefig(out_path+'/matrix.svg')
    plt.close()
    
    for i,c in enumerate(indices):
        min = indices[i]
        try:
            max = indices[i+1]
        except:
            max = len(matrix)
        plt.bar(range(len(row_nan_count.index[min:max])),row_nan_count.values[min:max])
        plt.ylim(0,1850)
        plt.xlabel('Genes')
        plt.ylabel('Missing number of data points')
        plt.savefig(out_path+'/row/0.row_dis'+chromosomes[i]+'.svg')
        plt.close()

    plt.bar(range(len(col_nan_count.index)),col_nan_count.values)
    plt.xlabel('Experiments')
    plt.ylabel('Missing number of data points')
    plt.savefig(out_path+'/col/0.col_dis.svg')
    plt.close()

print('plotted Nans', plot_nan)
np.nan_to_num(matrix,copy=False)
print('nans filled with 0')

#! Normalize data
matrix = normalize_2d(matrix)
print('Normalized matrix')
plot_sim_matrix(matrix,indices,chromosomes, save_loc=out_path)
print('plotting UMAP')
get_Umap(matrix,save_loc=out_path)

print('KNN impute')
df_impute = apply_KNN_impute(big_df,2)
df_impute.to_csv(path+'/imputed.csv')
# get the UMAP

print("plotting sim matrix, impute")
plot_sim_matrix(df_impute.to_numpy(),indices,chromosomes,"_impute",save_loc=out_path)
print("plotting UMAP, impute")
get_Umap(df_impute.to_numpy(),"_impute",save_loc=out_path)

print('Done')






# get pair wise similar


