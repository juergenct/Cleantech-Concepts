import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import yake
import spacy
import unicodedata
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Load Spacy model
nlp = spacy.load("en_core_web_lg")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to initialize YAKE model
language = "en"
max_ngram_size = 3
deduplication_threshold = 0.25
deduplication_algo = "seqm"
windowSize = 5
numOfKeywords = 25

yake_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                 dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                 features=None)

# Import test data
df = pd.read_json('/mnt/hdd01/patentsview/CPC Classification/df_cpc_y02_cleantech.json')

# Only keep columns 'cpc_classification', 'sequence', 'title', 'title_lower', 'full_title'
df = df[['cpc_classification', 'sequence', 'title', 'title_lower', 'full_title']]

# Initialize keywords_yake column
df['keywords_yake_title_lower'] = ''
df['keywords_yake_title_lower_noun_chunk'] = ''
df['noun_chunks'] = ''

# Iterate over rows in dataframe
for index, row in tqdm(df.iterrows()):
    try:
        if row['title_lower'] is None:
            df.at[index, 'keywords_yake_title_lower'] = [], []
            df.at[index, 'keywords_yake_title_lower_noun_chunk'] = []
        
        # Normalize the text with unicodedata
        row['title_lower'] = unicodedata.normalize("NFKD", row['title_lower']).encode('ASCII', 'ignore').decode('utf-8')
        # row['title_lower'] = re.sub(r"[^a-zA-Z- ]|^https?:\/\/.*[\r\n]*|\[.*?\]|\(.*?\)|\{.*?\}", " ", row['title_lower']).lower().strip()
        row['title_lower'] = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", row['title_lower'])
        row['title_lower'] = re.sub(r"https?:\/\/\S+", "", row['title_lower'])
        row['title_lower'] = re.sub(r"[^a-zA-Z- ]", " ", row['title_lower']).lower().strip()
        unfiltered_keywords = yake_extractor.extract_keywords(row['title_lower']) # Extract keywords using YAKE
        doc = nlp(row['title_lower'])
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        # Set dataframe to corresponding row in original dataframe
        df.at[index, 'keywords_yake_title_lower_noun_chunk'] = filtered_keywords
        df.at[index, 'noun_chunks'] = noun_chunks
    except:
        df.at[index, 'keywords_yake_title_lower_noun_chunk'] = [], []
        df.at[index, 'noun_chunks'] = []
    
# Save dataframe to json
df.to_json('/mnt/hdd01/patentsview/CPC Classification/df_keyword_y02_classification_noun_chunking.json', orient='records')
df.to_csv('/mnt/hdd01/patentsview/CPC Classification/df_keyword_y02_classification_noun_chunking.csv')
