import os
import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()

# Read in the data
df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_fulltext_cleantech.json', orient='records')

def process_row(row):
    claim_fulltext = ''
    for outerkey, innerdict in row['claims'].items():
        for innerkey, values in innerdict.items():
            if innerkey == 'claim_text':
                # Remove the claim number from the beginning of the claim text
                claim_text = re.sub(r'^\d+\.\s', ' ', values)
                claim_fulltext += claim_text
    # Return a Series with the processed claim_fulltext
    return pd.Series({'claim_fulltext': claim_fulltext})

# Apply the process_row function to each row of the DataFrame
df[['claim_fulltext']] = df.progress_apply(process_row, axis=1)

# Delete the 'claims' column
del df['claims']

# Save the DataFrame to json
df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_fulltext_cleantech.json', orient='records')