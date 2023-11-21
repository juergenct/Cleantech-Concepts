import os
import pandas as pd
import yake
from tqdm import tqdm
import spacy

# Load Spacy model
nlp = spacy.load("en_core_web_lg")

# Import test data
df = pd.read_json('/mnt/hdd01/patentsview/CPC Classification/df_cpc_y02_cleantech.json')

# Only keep columns 'cpc_classification', 'sequence', 'title', 'title_lower', 'full_title'
df = df[['cpc_classification', 'sequence', 'title', 'title_lower', 'full_title']]


# Specify custom parameters for YAKE
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.25
deduplication_algo = "seqm"
windowSize = 5
numOfKeywords = 25

# Initialize YAKE model
kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                    features=None)

# Initialize keywords_yake column
df['keywords_yake_title_lower'] = ''
df['keywords_yake_title_lower_noun_chunk'] = ''
df['noun_chunks'] = ''


# Iterate over rows in dataframe
for index, row in tqdm(df.iterrows()):
    # Extract keywords with YAKE
    unfiltered_keywords = kw_extractor.extract_keywords(row['title_lower'])
    df.at[index, 'keywords_yake_title_lower'] = unfiltered_keywords

    # Process text with Spacy
    doc = nlp(row['title_lower'])
    # Get noun chunks as a set for faster lookup
    noun_chunks = {chunk.text.strip().lower() for chunk in doc.noun_chunks}
  
    # Filter keywords based on whether they are contained within any of the noun_chunks
    filtered_keywords = [(keyword, score) for keyword, score in unfiltered_keywords if any(keyword in noun_chunk for noun_chunk in noun_chunks)]

    # Set dataframe to corresponding row in original dataframe
    df.at[index, 'keywords_yake_title_lower_noun_chunk'] = filtered_keywords

    df.at[index, 'noun_chunks'] = noun_chunks

# Save dataframe to json
df.to_json('/mnt/hdd01/patentsview/CPC Classification/df_keyword_y02_classification_noun_chunking.json', orient='records')
df.to_csv('/mnt/hdd01/patentsview/CPC Classification/df_keyword_y02_classification_noun_chunking.csv')
