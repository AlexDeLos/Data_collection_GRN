import GEOparse
import pandas as pd

from helpers import get_geo_list, mapping



geo_list = get_geo_list("data_addresses.csv")
df = pd.read_csv("genes_list.csv", index_col=0)
df_index = pd.read_csv("genes_list.csv", index_col=0)
first =True
duplicate_count = {}


for number,geo in enumerate(geo_list):
    print('stating ', number, geo)
    try:
        # time.sleep(5)
        gse = GEOparse.get_GEO(geo=geo, destdir="/tudelft.net/staff-umbrella/AT GE Datasets/data_",silent=True)
        print(df)
        print(gse)
        gse.metadata['type']

        key = list(gse.gpls)
        key = key[0]
        gpl_table=gse.gpls[key].table
        try:
            probe_to_gene_map = dict(zip(gpl_table["ID"], gpl_table["ORF"].map(mapping)))
        except Exception as error:
            print("---- ERROR MAKING MAP ----")
            print(error)
            continue
        for gsm_name, gsm in gse.gsms.items():
            print("Name: ", gsm_name)
            print ("Table data:",)
            print (gsm.table.head())
            in_df = gsm.table
            in_df['ID_REF'] = in_df['ID_REF'].map(probe_to_gene_map)
            # How to drop the numbers
            in_df = in_df.dropna()

            in_df.rename(columns={'VALUE': gsm_name}, inplace=True)
            in_df.set_index('ID_REF',inplace = True)
            # we take only the arabidopsis thaliana genges
            in_df.drop(in_df[~in_df.index.str.match(r'^At[1-5MC]g\d{5}$', case=False)].index, inplace=True) # we take only the a
            in_df = in_df.filter([gsm_name])

            if df.index.duplicated().any():
                print("Duplicates in df:", df[df.index.duplicated(keep=False)])

                # Create a dictionary to keep track of the count of duplicates
                duplicate_count = df.index.value_counts().to_dict()

                # Group by the index column and calculate the mean of the values
                df = df.groupby('ID_REF')[in_df.columns[0]].mean().reset_index() # assuming there 

            if in_df.index.duplicated().any():
                print("Duplicates in df:", in_df[in_df.index.duplicated(keep=False)])

                # Create a dictionary to keep track of the count of duplicates
                duplicate_count = in_df.index.value_counts().to_dict()

                # Group by the index column and calculate the mean of the values
                in_df = in_df.groupby('ID_REF')[in_df.columns[0]].mean().reset_index() # assuming there is only one value

            try:
                if in_df.index.name != 'ID_REF':
                    in_df.set_index('ID_REF',inplace = True)
                # Fill in all the nans
                complete_in = pd.concat([df_index, in_df], axis=1)
                # complete_in = complete_in.transform(lambda x: x.fillna(x.mean()))
                df = pd.concat([df, complete_in], axis=1)
                x = 0
            except Exception as error:
                print(error)
                pass
            
    except Exception as error:
        print(error)
        print("-----An error occured, probably an empty dataframe")
        
    if number %10 == 0 and number != 0:
        df.to_csv("/tudelft.net/staff-umbrella/AT GE Datasets/df_nan/df_"+str(number)+".csv")
        df = pd.DataFrame(index=df.index)
    
df.to_csv("/tudelft.net/staff-umbrella/AT GE Datasets/df_nan/df_last.csv")
df = pd.DataFrame(index=df.index)


#! For now we ignore GPLs
# print()
# print("GPL example:")
# for gpl_name, gpl in gse.gpls.items():
#     print("Name: ", gpl_name)
#     print("Metadata:",)
#     for key, value in gpl.metadata.items():
#         print(" - %s : %s" % (key, ", ".join(value)))
#     print("Table data:",)
#     print(gpl.table.head())
#     break