import os
import pandas as pd
import yake

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')

# Concatenate abstracts for same value in 'cpc_subgroup' column - SHOULD I REALLY DO THIS???
df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# # Remove duplicate rows on 'cpc_subgroup' column
# df = df.drop_duplicates(subset=['cpc_subgroup'])

# Specify custom parameters
language = "en"
max_ngram_size = 3
dedulication_threshold = 0.9
deduplication_algo = "seqm"
windowSize = 1
numOfKeywords = 410

# Initialize YAKE model
kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=dedulication_threshold,
                                    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                    features=None)

# Iterate over rows in dataframe
for index, row in df.iterrows():
    # Extract keywords
    keywords = kw_extractor.extract_keywords(row['patent_abstract'])
    # Create empty list
    keywords_list = []
    # Iterate over keywords
    for kw in keywords:
        # Append keywords to list
        keywords_list.append(kw[0])
    # Create new column with keywords
    df.loc[index, 'keywords'] = ', '.join(keywords_list)

# Save dataframe to json
df.to_json('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/YAKE/df_sample_keyphrase.json')