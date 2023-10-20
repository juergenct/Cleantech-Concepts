import os
import pandas as pd
from keybert import KeyBERT

# Import data
df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Keybert/g_patent_claims_cleantech_test.json')

# Concatenate abstracts for same value in 'cpc_subgroup' column 
# df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# # Remove duplicate rows on 'cpc_subgroup' column
# df = df.drop_duplicates(subset=['cpc_subgroup'])

# Initialize KeyBERT model
# kw_model = KeyBERT(model='climatebert/distilroberta-base-climate-f')
kw_model =KeyBERT(model='AI-Growth-Lab/PatentSBERTa')

mmr_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Iterate over rows in dataframe
# Try out with MMR and rather high diversity 0.7, might try with lower diversity
# Parameter Loop for MMR diversity
for mmr in mmr_values:
    column_name = 'keywords_mmr_' + str(mmr)
    for index, row in df.iterrows():
        # Extract keywords
        keywords = kw_model.extract_keywords(row['claim_fulltext'],keyphrase_ngram_range=(2, 3),stop_words="english", use_mmr=True, diversity=0.7, top_n=10)
        # Create empty list
        keywords_list = []
        # Iterate over keywords
        for kw in keywords:
            # Append keywords to list
            keywords_list.append(kw[0])
        # Create new column with keywords
        df.loc[index, column_name] = ', '.join(keywords_list)

# Save dataframe to json
df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Keybert/g_patent_claims_cleantech_keybert_test.json', orient='records')