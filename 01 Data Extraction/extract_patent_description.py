import os
import csv
import pandas as pd
from tqdm import tqdm

# Function to process chunks
def process_chunk(chunk, df_patent_id):
    chunk['patent_id'] = chunk['patent_id'].astype(str)
    merged_chunk = chunk.merge(df_patent_id, on='patent_id', how='right', validate='many_to_one')
    merged_chunk.dropna(subset=['description_text'], inplace=True)
    return merged_chunk


# Load Patent Data
df_patent_id = pd.read_csv('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_ids_patentsview_cleantech.csv')
df_patent_id['patent_id'] = df_patent_id['patent_id'].astype(str)

# Get all .tsv.zip files in directory
path = '/mnt/hdd01/patentsview/Description/'
files_tsv = [os.path.join(path, file) for file in os.listdir(path) if '.tsv.zip' in file]
file_years = [file.rsplit('_', 1)[-1].replace('.tsv.zip', '') for file in files_tsv]

# Only include the following years
# incl_years = ['2018']

# Include the files by incl_years
# files_tsv = [file for file in files_tsv if file_years[files_tsv.index(file)] in incl_years]

for file in tqdm(files_tsv):
    try:
        df_patent_desc_list = []
        df_patent_desc_temp = pd.read_csv(file, compression='zip', sep='\t', on_bad_lines='warn') #, quoting=csv.QUOTE_NONNUMERIC)
        df_patent_desc_list.append(process_chunk(df_patent_desc_temp, df_patent_id))
        # Concatenate all df_patent_desc in df_patent_desc
        df_patent_desc = pd.concat(df_patent_desc_list)
        # Group and Create nested dictionary structure
        df_patent_desc_grouped = df_patent_desc.groupby('patent_id').apply(lambda x: {'description_text': x['description_text'].iloc[0], 'description_length': x['description_length'].iloc[0]}).reset_index(name='description')
        # Save to json
        df_patent_desc_grouped.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_description_cleantech_' + str(file.rsplit('_', 1)[-1].replace('.tsv.zip', '')) + '.json', orient='records')
    except:
        print(file)
        continue