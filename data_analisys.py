import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from inmoose.pycombat import pycombat_norm
from sklearn.preprocessing import RobustScaler

from helpers import get_first_indexs,plot_sim_matrix,get_Umap, normalize_2d,apply_KNN_impute,hierarchical_clustering



plot_nan = True

# Get CSV files list from a folder
path = '/tudelft.net/staff-umbrella/AT GE Datasets/processed_data'
out_path = '/tudelft.net/staff-umbrella/AT GE Datasets/figures'

# path = 'df_processed'
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
            max_var = indices[i+1]
        except:
            max_var = len(matrix)
        plt.bar(range(len(row_nan_count.index[min:max_var])),row_nan_count.values[min:max_var])
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


# With simple replacement
print('plotted Nans', plot_nan)
np.nan_to_num(matrix,copy=False)
print('nans filled with 0')


matrix = big_df.to_numpy()

def get_study(sample: str):
    return int(sample.split('_')[-1])
study_map = list(map(get_study,big_df.columns))

def get_method(sample: str):
    return str(sample.split('_')[1])
methods = set(map(get_method,big_df.columns))


print('plotting UMAP')
get_Umap(matrix.T,name='_samples',study_map=study_map,save_loc=out_path, title='Samples coloured by study (No impute)')
get_Umap(matrix,name='_genes',save_loc=out_path, title='Gene expression clusters (No impute)')


# With KNN impute

# here the matrix veriable needs to be reset 
#! Reset matrix
matrix = None
big_df = big_df.loc[:, ~(big_df > 1000).any()]
# KNN Impute
try: 
    print('reading file from: ' + path+'/imputed.csv')
    df_impute = pd.read_csv(path+'/imputed.csv', index_col=0)
except FileNotFoundError:
    print('Could not read file, running KNN impute')
    df_impute = apply_KNN_impute(big_df,5)
    print('KNN impute ran, saving file')
    df_impute.to_csv(path+'/imputed.csv')
    print('file saved at: ' + path+'/imputed.csv')
    # get the UMAP


# NORMALIZE
#? Apply batch correction

# https://inmoose.readthedocs.io/en/latest/pycombatnorm.html
# pycombat_norm
# plt.boxplot(df_impute)
# plt.savefig(out_path+'/box_plot_pre.svg')
# plt.close()

study_map = list(map(get_study,df_impute.columns))
df_corrected = pycombat_norm(df_impute, study_map) #! TODO: this needs the nans removed before we can run it. maybe run impute before or out this before the mapping
df_corrected.to_csv(path+'/corrected.csv')

normalized_df = (df_corrected - df_corrected.min()) / (df_corrected.max() - df_corrected.min())

standardized_df = (df_corrected - df_corrected.mean()) / df_corrected.std()

scaler = RobustScaler()
robust_df = pd.DataFrame(scaler.fit_transform(df_corrected), columns=df_corrected.columns)

matrices = [normalized_df.to_numpy(),standardized_df.to_numpy(),robust_df.to_numpy()]



for i,mat in enumerate(matrices):
    print('starting plot')
    plt.imshow(mat, cmap='hot')
    plt.colorbar()
    plt.savefig(out_path+'/impute_matrix'+str(i)+'.svg')
    plt.close()
    plt.hist(mat,bins=1000 )
    plt.savefig(out_path+'/histo_'+str(i)+'.svg')
    plt.close()


impute_matrix = df_corrected.to_numpy()
plt.boxplot(df_corrected)
plt.savefig(out_path+'/box_plot_post.svg')
plt.close()

print('plotting sim matrix, impute')
plot_sim_matrix(impute_matrix,indices,chromosomes,'_impute',save_loc=out_path)
print('plotting UMAP, impute')
get_Umap(impute_matrix.T,name='_samples_impute',study_map=study_map,save_loc=out_path, title='Samples coloured by study (impute)')
get_Umap(impute_matrix,name='_genes_impute',save_loc=out_path, title='Gene expression clusters (Impute)')


hierarchical_clustering(impute_matrix)
print('Done')






# get pair wise similar


