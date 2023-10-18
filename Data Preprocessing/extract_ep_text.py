import pandas as pd
import glob
import os
from tqdm import tqdm

# Read in the data
df_epo_cleantech_granted = pd.read_csv('/mnt/hdd01/PATSTAT Working Directory/cleantech_ep_granted.csv')

# Cast appln_id to string
df_epo_cleantech_granted['appln_id'] = df_epo_cleantech_granted['appln_id'].astype(str)

# Get all txt files from directory "/mnt/hdd01/EP_Fulltext_Data/" that start with EP and end with .txt
path = '/mnt/hdd01/EP_Fulltext_Data/'
txt_files = glob.glob(path + "EP*.txt")

# Create empty result dataframe
df_epo_cleantech_result = pd.DataFrame()

# Loop through all txt files
for txt_file in tqdm(txt_files):
    print("Reading in " + txt_file + " ...")
    df_txt = pd.read_csv(txt_file, sep='\t', header=0, names=['appln_auth', 'appln_id', 'appln_kind', 'appln_date', 'appln_lng', 'appln_comp', 'appln_text_type', 'appln_text'])

    # Cast appln_id to string
    df_txt['appln_id'] = df_txt['appln_id'].astype(str)

    # Print first 5 rows of df_txt['appln_id']
    print(df_txt['appln_id'].head())

    # Merge df_txt and df_epo_cleantech_granted on appln_id, right join, many-to-many
    df_txt = pd.merge(df_txt, df_epo_cleantech_granted, how='right', on='appln_id', validate='many_to_many')

    # Delete all rows where appln_text is NaN
    df_txt = df_txt.dropna(subset=['appln_text'])

    # Append df_txt to df_epo_cleantech_result
    df_epo_cleantech_result = pd.concat([df_epo_cleantech_result, df_txt], ignore_index=True)

    # Print out how many rows df_epo_cleantech_result has
    print("Successfully extracted " + str(df_txt['appln_id'].nunique()) + " patents.")

# Write df_epo_cleantech_result to csv
df_epo_cleantech_result.to_csv('/mnt/hdd01/PATSTAT Working Directory/cleantech_epo_cleantech_text_data.csv ')