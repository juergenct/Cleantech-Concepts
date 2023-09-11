import os
import pandas as pd
from rake_nltk import Rake

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')

# Concatenate abstracts for same value in 'cpc_subgroup' column - SHOULD I REALLY DO THIS???
df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# # Remove duplicate rows on 'cpc_subgroup' column
# df = df.drop_duplicates(subset=['cpc_subgroup'])

# Initialize RAKE model
rake_extractor = Rake()

# Iterate over rows in dataframe
for index, row in df.iterrows():
    # Extract keywords
    rake_extractor.extract_keywords_from_text(row['patent_abstract'])
    keywords = rake_extractor.get_ranked_phrases()
    
    # Create new column with keywords
    df.loc[index, 'keywords'] = ', '.join(keywords)

# Save dataframe to json
df.to_json('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/RAKE/df_sample_keyphrase.json')