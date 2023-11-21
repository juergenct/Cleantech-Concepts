import os
import pandas as pd
from tqdm import tqdm

# Load Fulltext Data
# Get all .tsv files in directory
path = '/mnt/hdd01/patentsview/Claims/'
files = os.listdir(path)
files_tsv = [f for f in files if f[-3:] == 'tsv']

# Load Patent Data
df_patent_id = pd.read_csv('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_ids_patentsview_cleantech.csv')
df_patent_id['patent_id'] = df_patent_id['patent_id'].astype(str)
df_patent_claims_list = []

for file in tqdm(files_tsv):
    df_claims = pd.read_csv(path + file, sep='\t', header=0)
    df_claims['patent_id'] = df_claims['patent_id'].astype(str)
    # Sort df_claims by patent_id and then by claim_sequence
    df_claims.sort_values(by=['patent_id', 'claim_sequence'], inplace=True)
    # Match df_patent_id and df_claims on patent_id
    df_patent_claims = df_claims.merge(df_patent_id, on='patent_id', how='right', validate='many_to_one')
    # Delete Rows with only Null value in 'claim_sequence'
    df_patent_claims.dropna(subset=['claim_sequence'], inplace=True)
    # Concatenate df_patent_claims to df_patent_claims_list
    df_patent_claims_list.append(df_patent_claims)

# Concatenate all df_patent_claims in df_patent_claims
df_patent_claims = pd.concat(df_patent_claims_list)

# Group by patent_id
df_patent_claims_grouped = df_patent_claims.groupby('patent_id').agg({
    'claim_sequence': list,
    'claim_text': list,
    'dependent': list,
    'claim_number': list,
    'exemplary': list,
}).reset_index()

# Create dictionary of claims
df_patent_claims_grouped['claims'] = df_patent_claims_grouped.apply(lambda row: {i+1: {'claim_sequence': claim_sequence, 'claim_text': claim_text, 'dependent': dependent, 'claim_number': claim_number, 'exemplary': exemplary} for i, (claim_sequence, claim_text, dependent, claim_number, exemplary) in enumerate(zip(row['claim_sequence'], row['claim_text'], row['dependent'], row['claim_number'], row['exemplary']))}, axis=1)

# Drop original columns
df_patent_claims_grouped.drop(['claim_sequence', 'claim_text', 'dependent', 'claim_number', 'exemplary'], axis=1, inplace=True)

# Save to json
df_patent_claims_grouped.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_cleantech.json', orient='records')