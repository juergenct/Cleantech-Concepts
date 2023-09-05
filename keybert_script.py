import os
import pandas as pd
from keybert import KeyBERT

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')

# Concatenate abstracts for same value in 'cpc_subgroup' column
df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# Remove duplicate rows on 'cpc_subgroup' column
df = df.drop_duplicates(subset=['cpc_subgroup'])

# Initialize YAKE model
kw_model = KeyBERT(model='climatebert/distilroberta-base-climate-f')

# Iterate over rows in dataframe
# Try out with MMR and rather high diversity 0.7, might try with lower diversity
for index, row in df.iterrows():
    # Extract keywords
    keywords = kw_model.extract_keywords(row['patent_abstract'],keyphrase_ngram_range=(1, 3),stop_words="english", use_mmr=True, diversity=0.7)
    # Create empty list
    keywords_list = []
    # Iterate over keywords
    for kw in keywords:
        # Append keywords to list
        keywords_list.append(kw[0])
    # Create new column with keywords
    df.loc[index, 'keywords'] = ', '.join(keywords_list)

# Save dataframe to json
df.to_json('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/KeyBERT/df_sample_keyphrase.json')