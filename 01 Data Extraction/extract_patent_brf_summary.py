import os
import pandas as pd
from tqdm import tqdm

# Load Fulltext Data
# Get all .tsv files in directory
path = '/mnt/hdd01/patentsview/Brief Summary/'
files = os.listdir(path)
files_tsv = [f for f in files if f[-3:] == 'tsv']

# Load Patent Data
df_patent_id = pd.read_csv('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_ids_patentsview_cleantech.csv')
df_patent_id['patent_id'] = df_patent_id['patent_id'].astype(str)
df_patent_brf_summary_list = []

for file in tqdm(files_tsv):
    df_brf_summary = pd.read_csv(path + file, sep='\t', header=0)
    df_brf_summary['patent_id'] = df_brf_summary['patent_id'].astype(str)
    # Sort df_brf_summary by patent_id 
    df_brf_summary.sort_values(by=['patent_id'], inplace=True)
    # Match df_patent_id and df_brf_summary on patent_id
    df_patent_brf_summary = df_brf_summary.merge(df_patent_id, on='patent_id', how='right', validate='many_to_one')
    # Delete Rows with only Null value in 'summary_text'
    df_patent_brf_summary.dropna(subset=['summary_text'], inplace=True)
    # Concatenate df_patent_brf_summary to df_patent_brf_summary_list
    df_patent_brf_summary_list.append(df_patent_brf_summary)

# Concatenate all df_patent_brf_summary in df_patent_brf_summary
df_patent_brf_summary = pd.concat(df_patent_brf_summary_list)

# Save to json
df_patent_brf_summary.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_brf_summary_cleantech.json', orient='records')