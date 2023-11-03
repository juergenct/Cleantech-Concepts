import os
import pandas as pd
# import modin.pandas as pd
import yake
from tqdm import tqdm

# Import test data
df = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/Reliance on Science/cleantech_epo_rel_on_science_abstract.json')

# Drop column 'keywords_yake'
# df = df.drop(columns=['keywords_yake'])

# Concatenate 'patent_title' and 'patent_abstract' columns
# df['patent_title_abstract'] = df['patent_title'] + ' ' + df['patent_abstract']

# Specify custom parameters
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.25
deduplication_algo = "seqm"
windowSize = 5
numOfKeywords = 10

# Initialize YAKE model
kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                    features=None)

# Initialize keywords_yake column
df['keywords_yake_abstract'] = ''

# Iterate over rows in dataframe
for index, row in tqdm(df.iterrows()):
    # Extract keywords
    keywords = kw_extractor.extract_keywords(row['abstract'])
    # Set dataframe to corresponding row in original dataframe
    df.at[index, 'keywords_yake_abstract'] = [keywords]

# Save dataframe to json
df.to_json('/mnt/hdd01/PATSTAT Working Directory/Reliance on Science/cleantech_epo_rel_on_science_abstract_yake.json', orient='records')