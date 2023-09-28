import pandas as pd
from sqlalchemy import create_engine, URL
from tqdm import tqdm

url_object = URL.create(
    drivername='postgresql+psycopg2',
    username='tie',
    password='TIE%2023!tuhh',
    host='134.28.58.100',
    port=45432,
    database='openalex_db',
)

# Create engine
engine = create_engine(url_object)
# Check if connection is successful
engine.connect()

# Load dataframe containing OpenAlex IDs
df = pd.read_csv('/mnt/hdd01/patentsview/Reliance on Science - Cleantech Patents/df_oaid_Cleantech_Y02.csv')

# Convert column oaid and patent_id to string
df['oaid'] = df['oaid'].astype(str)
df['patent_id'] = df['patent_id'].astype(str)

# Erase everything in oaid and patent_id after the dot
df['oaid'] = df['oaid'].str.split('.').str[0]
df['patent_id'] = df['patent_id'].str.split('.').str[0]

# Add column 'full_oaid' with https://openalex.org/W + oaid
df['full_oaid'] = 'https://openalex.org/W' + df['oaid']

# Keep only columns oaid, patent_id and full_oaid
df = df[['oaid', 'patent_id', 'full_oaid']]

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

# Iterate over abstract_inverted_index column
for index, row in tqdm(result.iterrows()):
    word_index = []
    try:
        for key, value in row['abstract_inverted_index'].items():
            if key == 'InvertedIndex':
                for innerkey, innervalue in value.items():
                    for index in innervalue:
                        word_index.append([innerkey, index])
        # Sort list by index
        word_index.sort(key=lambda x: x[1])
        # Join first element of each list in word_index
        abstract = ' '.join([i[0] for i in word_index])
        # Add column abstract to result dataframe
        result.loc[index, 'abstract'] = abstract
    except AttributeError:
        continue

# Save result dataframe to json
result.to_json('/mnt/hdd01/patentsview/Reliance on Science - Cleantech Patents/df_oaid_Cleantech_Y02_works.json', orient='records')