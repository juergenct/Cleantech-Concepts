import os
import pandas as pd
from tqdm import tqdm

# Load Fulltext Data
# Get all .tsv files in directory
path = 'path/to/data/Description'
files = os.listdir(path)
files_tsv = [f for f in files if f[-3:] == 'tsv']

# Load Patent Data
df_patent_id = pd.read_csv('path/to/data/g_patent_ids_rel_on_science_cleantech.csv')
df_patent_id['patent_id'] = df_patent_id['patent_id'].astype(str)
df_patent_desc_list = []

for file in tqdm(files_tsv):
    df_desc = pd.read_csv(path + file, sep='\t', header=0)
    df_desc['patent_id'] = df_desc['patent_id'].astype(str)
    # Sort df_desc by patent_id
    df_desc.sort_values(by=['patent_id'], inplace=True)
    # Match df_patent_id and df_desc on patent_id
    df_patent_desc = df_desc.merge(df_patent_id, on='patent_id', how='right', validate='many_to_one')
    # Delete Rows with only Null value in 'description_text'
    df_patent_desc.dropna(subset=['description_text'], inplace=True)
    # Concatenate df_patent_desc to df_patent_desc_list
    df_patent_desc_list.append(df_patent_desc)

# Concatenate all df_patent_desc in df_patent_desc
df_patent_desc = pd.concat(df_patent_desc_list)

# Add nested dictionary structure
df_patent_desc_grouped = df_patent_desc.groupby('patent_id').apply(lambda x: {'description_text': x['description_text'].iloc[0], 'description_length': x['description_length'].iloc[0]}).reset_index(name='description')

# Save to json
df_patent_desc_grouped.to_json('path/to/data/g_detail_desc_text_2000.json', orient='records')