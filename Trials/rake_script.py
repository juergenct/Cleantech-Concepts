import os
import pandas as pd
from rake_nltk import Rake

# Import test data
df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Rake/g_patent_claims_cleantech_test.json')
# Concatenate abstracts for same value in 'cpc_subgroup' column - SHOULD I REALLY DO THIS???
# df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# # Remove duplicate rows on 'cpc_subgroup' column
# df = df.drop_duplicates(subset=['cpc_subgroup'])

# Initialize RAKE model
rake_extractor = Rake()

# Iterate over rows in dataframe
for index, row in df.iterrows():
    # Extract keywords
    rake_extractor.extract_keywords_from_text(row['claim_fulltext'])
    keywords = rake_extractor.get_ranked_phrases()
    
    # Create new column with keywords
    df.loc[index, 'keywords'] = ', '.join(keywords)

# Save dataframe to json
df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Rake/g_patent_claims_cleantech_rake_test.json', orient='records')