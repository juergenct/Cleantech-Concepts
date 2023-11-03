import pandas as pd
import os
from sqlalchemy import create_engine, URL
from tqdm import tqdm

url_object = URL.create(
    drivername='postgresql+psycopg2',
    username='tie',
    password='TIE%2023!tuhh',
    # host='134.28.58.100',
    # host='tie-workstation.tail6716.ts.net',
    host='localhost',
    port=45432,
    database='openalex_db',
)

# Create engine
engine = create_engine(url_object)
# Check if connection is successful
engine.connect()

# Load dataframe containing OpenAlex IDs
df = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/Reliance on Science/cleantech_epo_rel_on_science_agg.json')

# Convert column oaid and patent_id to string
df['oaid'] = df['oaid'].astype(str)
df['publn_nr'] = df['publn_nr'].astype(str)

# Erase everything in oaid and patent_id after the dot
df['oaid'] = df['oaid'].str.split('.').str[0]
df['publn_nr'] = df['publn_nr'].str.split('.').str[0]

# Add column 'full_oaid' with https://openalex.org/W + oaid
df['full_oaid'] = 'https://openalex.org/W' + df['oaid']

# Keep only columns oaid, patent_id and full_oaid
df = df[['oaid', 'publn_nr', 'full_oaid']]

# Insert temporary table into database
df.to_sql('temp_table', engine, if_exists='replace', index=False)

# SQL query to get all data from data table that has an id in the temp_table
query = '''
    SELECT *
    FROM openalex.works 
    JOIN temp_table ON openalex.works.id = temp_table.full_oaid
'''

# Execute query
result = pd.read_sql(query, engine)

# Delete all empty rows in column abstract_inverted_index with value None
result_filtered = result[result['abstract_inverted_index'].notna()]

print(f"Deleted {len(result) - len(result_filtered)} rows without abstract_inverted_index")

# Iterate over abstract_inverted_index columnc
for index, row in tqdm(result_filtered.iterrows()):
    word_index = []
    try:
        for key, value in row['abstract_inverted_index'].items():
            if key == 'InvertedIndex':
                for innerkey, innervalue in value.items():
                    for innerindex in innervalue:
                        word_index.append([innerkey, innerindex])
        # Sort list by index
        word_index.sort(key=lambda x: x[1])
        # Join first element of each list in word_index
        abstract = ' '.join([i[0] for i in word_index])
        # Add column abstract to result dataframe
        # result.loc[index, 'abstract'] = abstract
        result_filtered.at[index, 'abstract'] = abstract
        # print(result.loc[index, 'abstract'])
    except AttributeError:
        continue

# Print Result
print(f"Succesfully processed {len(result_filtered)} rows from {len(df)} rows in original dataframe")

# Save result dataframe to json
result_filtered.to_json('/mnt/hdd01/PATSTAT Working Directory/Reliance on Science/cleantech_epo_rel_on_science_abstract.json', orient='records')