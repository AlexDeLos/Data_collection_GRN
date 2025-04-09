import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from inmoose.pycombat import pycombat_norm
from sklearn.preprocessing import RobustScaler

from helpers import get_first_indexs,plot_sim_matrix,get_Umap, apply_KNN_impute,hierarchical_clustering,box_plot



plot_nan = True
plot_Umap = True
plot_boxPlots = True
plot_simMatrix = True

# Get CSV files list from a folder
path = '/tudelft.net/staff-umbrella/AT GE Datasets/df_local/df'
out_path = '/tudelft.net/staff-umbrella/AT GE Datasets/figures'

norm_dic = {
    0: 'Nromalize',
    1: 'Stardardize',
    2: 'Robust'
}

# path = 'df'
# out_path = 'figures'
try:
    filtered_df = pd.read_csv(path+'/filter.csv', index_col=0)
    print('succesfully loaded data')
except:
    csv_files = glob.glob(path + '/df*')

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

#! Remove columns from the datafram
#TODO: static
# big_df.loc[:, ~(big_df > 1000000).any()]
big_df = big_df.filter(regex=r'^(?!.*GSM463683).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463684).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463685).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463686).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463687).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463688).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463689).*$')
big_df = big_df.filter(regex=r'^(?!.*GSM463690).*$')


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
        min_var = indices[i]
        try:
            max_var = indices[i+1]
        except:
            max_var = len(matrix)
        plt.bar(range(len(row_nan_count.index[min_var:max_var])),row_nan_count.values[min_var:max_var])
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



def get_study(sample: str):
    return int(sample.split('_')[-1])
study_map = list(map(get_study,big_df.columns))

def get_method(sample: str):
    return str(sample.split('_')[1])

if plot_Umap:
    matrix = big_df.to_numpy()

    methods = set(map(get_method,big_df.columns))
    print('plotting UMAP')
    get_Umap(matrix.T,name='_samples',study_map=study_map,save_loc=out_path, title='Samples coloured by study (No impute)')
    get_Umap(matrix,name='_genes',save_loc=out_path, title='Gene expression clusters (No impute)')
    matrix = None


# With KNN impute

# here the matrix veriable needs to be reset 
#! Reset matrix
# big_df = big_df.loc[:, ~(big_df > 1000).any()]
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
if plot_boxPlots:
    box_plot(df_impute,100, out_path+'/box_uncorrected_plots/')
if plot_Umap:
    get_Umap(df_impute.to_numpy().T,name='_samples_impute',study_map=study_map,save_loc=out_path, title='Samples coloured by study')
    get_Umap(df_impute.to_numpy(),name='_genes_impute',save_loc=out_path, title='Gene expression clusters')

try:
    df_corrected = pd.read_csv(path+'/corrected.csv',index_col=0)
except FileNotFoundError:
    study_map = list(map(get_study,df_impute.columns))
    raise ValueError("Running in the cluster")
    # df_corrected = pycombat_norm(df_impute, study_map) #! TODO: this needs the nans removed before we can run it. maybe run impute before or out this before the mapping
    # df_corrected.to_csv(path+'/corrected.csv')

hierarchical_clustering(df_corrected,path=out_path, name='CorrectedS')

if plot_boxPlots:
    box_plot(df_corrected,100, out_path+'/box_corrected_plots/')
if plot_Umap:
    get_Umap(df_corrected.to_numpy().T,name='_samples_impute_corrected',study_map=study_map,save_loc=out_path, title='Samples coloured by study')
    get_Umap(df_corrected.to_numpy(),name='_genes_impute_corrected',save_loc=out_path, title='Gene expression clusters')

if plot_simMatrix:
    plot_sim_matrix(df_impute,indices,chromosomes,name='_Gene_impute_no_correction',save_loc=out_path,title='Gen Sim (Not Corrected)')
    plot_sim_matrix(df_impute.T,name='_Sample_impute_no_correction',save_loc=out_path,title='Sample Sim (Not Corrected)')
    plot_sim_matrix(df_corrected,indices,chromosomes,name='_Gene_impute_corrected(No_norm)',save_loc=out_path,title='Gene Sim (Corrected)')
    plot_sim_matrix(df_corrected.T,name='_Sample_impute_corrected(No_norm)',save_loc=out_path,title='Sample Sim (Corrected)')

normalized_df = (df_corrected - df_corrected.min()) / (df_corrected.max() - df_corrected.min())

standardized_df = (df_corrected - df_corrected.mean()) / df_corrected.std()

scaler = RobustScaler()
robust_df = pd.DataFrame(scaler.fit_transform(df_corrected), columns=df_corrected.columns)

print('Saving normalized:')
normalized_df.to_csv(path+'/normalized.csv')
standardized_df.to_csv(path+'/standardized.csv')
robust_df.to_csv(path+'/robust.csv')

normalized_dfs = [normalized_df,standardized_df,robust_df]


for i,normalized_df_entry in enumerate(normalized_dfs):
    norm_matrix = normalized_df_entry.to_numpy()
    if plot_boxPlots:
        box_plot(normalized_df_entry,100, out_path+'/box_norm_plots/', group=i)
    if plot_simMatrix:
        print('plotting sim matrix, impute')
        plot_sim_matrix(norm_matrix,indices,chromosomes,name='_Gene_impute_'+norm_dic[i],save_loc=out_path, title='Gene Sim ('+norm_dic[i]+')')
        plot_sim_matrix(norm_matrix.T,name='_Sample_impute_'+norm_dic[i],save_loc=out_path, title='Sample Sim ('+norm_dic[i]+')')

    if plot_Umap:
        print('plotting UMAP, impute')
        get_Umap(norm_matrix,name='_genes_final_'+norm_dic[i]+'_',save_loc=out_path, title='Gene expression clusters ('+norm_dic[i]+')')
        get_Umap(norm_matrix.T,name='_samples_final_'+norm_dic[i]+'_',study_map=study_map,save_loc=out_path, title='Samples coloured by study ('+norm_dic[i]+')')


    hierarchical_clustering(norm_matrix,path=out_path, name=norm_dic[i])

print('Done')
