import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get CSV files list from a folder
path = 'df'
csv_files = glob.glob(path + "/*.csv")

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file, index_col=0) for file in csv_files)

# Concatenate all DataFrames
big_df = pd.concat(df_list, axis=1)


print(big_df)
row_nan_count = big_df.isna().sum(axis=1)
filter_row = big_df.isna().sum(axis=1)>1000
col_nan_count = big_df.isna().sum(axis=0)

matrix_nan = big_df.isna().to_numpy()

plt.imshow(matrix_nan[:10000], cmap='hot', interpolation='nearest')
plt.savefig('matrix.svg')
plt.close()



plt.bar(range(len(row_nan_count.index)),row_nan_count.values)
plt.savefig('row_dis.svg')
plt.close()

plt.bar(range(len(col_nan_count.index)),col_nan_count.values)
plt.savefig('col_dis.svg')
plt.close()

