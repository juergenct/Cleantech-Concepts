import os
import pandas as pd
import yake
from tqdm import tqdm

# Import test data
df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_fulltext_cleantech.json')

# Concatenate 'patent_title' and 'patent_abstract' columns
# df['patent_title_abstract'] = df['patent_title'] + ' ' + df['patent_abstract']

# Concatenate abstracts for same value in 'cpc_subgroup' column
# df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# # Remove duplicate rows on 'cpc_subgroup' column
# df = df.drop_duplicates(subset=['cpc_subgroup'])

# Specify custom parameters
language = "en"
max_ngram_size = 3
dedulication_threshold = 0.25
deduplication_algo = "seqm"
windowSize = 5
numOfKeywords = 10

# Initialize YAKE model
kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=dedulication_threshold,
                                    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                    features=None)

# Iterate over rows in dataframe
for index, row in tqdm(df.iterrows()):
    # Extract keywords
    keywords = kw_extractor.extract_keywords(row['claim_fulltext'])
    # Create empty list
    keywords_list = []
    # Iterate over keywords
    for kw in keywords:
        # Append keywords to list
        keywords_list.append(kw[0])
    # Create new column with keywords
    df.loc[index, 'keywords_yake'] = ', '.join(keywords_list)

# Save dataframe to json
df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_cleantech_yake.json', orient='records')