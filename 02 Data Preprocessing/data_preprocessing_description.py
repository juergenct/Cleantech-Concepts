import os
import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()

# Read in the data
df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_description_cleantech.json', orient='records')

def process_row(row):
    description_fulltext = ''
    for key, value in row['description'].items():
        if key == 'description_text':
            description_fulltext = value
    # Return a Series with the processed claim_fulltext
    return pd.Series({'description_text': description_fulltext})

# Apply the process_row function to each row of the DataFrame
df[['description_text']] = df.progress_apply(process_row, axis=1)

# Save the DataFrame to json
df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_description_fulltext_cleantech.json', orient='records')