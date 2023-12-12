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

# Function to extract keywords and filter with noun chunks
def extract_and_filter(row):
    try:
        if row['cleaned_claims'] is None:
            print("No claims found for patent number: " + str(row['publn_nr']))
            return [], []
        
        # Normalize the text with unicodedata
        row['cleaned_claims'] = unicodedata.normalize("NFKD", row['cleaned_claims']).encode('ASCII', 'ignore').decode('utf-8')
        # row['cleaned_claims'] = re.sub(r"[^a-zA-Z- ]|^https?:\/\/.*[\r\n]*|\[.*?\]|\(.*?\)|\{.*?\}", " ", row['cleaned_claims']).lower().strip()
        row['cleaned_claims'] = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", row['cleaned_claims'])
        row['cleaned_claims'] = re.sub(r"https?:\/\/\S+", "", row['cleaned_claims'])
        row['cleaned_claims'] = re.sub(r"[^a-zA-Z- ]", " ", row['cleaned_claims']).lower().strip()
        unfiltered_keywords = yake_extractor.extract_keywords(row['cleaned_claims']) # Extract keywords using YAKE
        doc = nlp(row['cleaned_claims'])
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        return unfiltered_keywords, filtered_keywords
    except:
        print("Error in patent number: " + str(row['publn_nr']))
        return [], []

def main():
    # Import test data
    # df = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/cleantech_epo_text_data_pivot_cleaned.json')
    df = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_epo_non_cleantech_text_data_pivot_claims_cleaned.json')

    # Drop all columns except 'publn_nr', 'TITLE', 'appln_id', 'cpc_class_symbol', 'cleaned_claims'
    # df = df[['publn_nr', 'TITLE', 'appln_id', 'cpc_class_symbol', 'cleaned_claims']]

    # Cast column 'cleaned_claims' to string
    df['cleaned_claims'] = df['cleaned_claims'].astype(str)

    # Set up multiprocessing
    num_cores = min(6, cpu_count())
    pool = Pool(num_cores)

    # Apply the function in parallel
    results = list(tqdm(pool.imap(extract_and_filter, [row for _, row in df.iterrows()]), total=len(df)))

    # Split results into separate columns
    df['keywords_yake_claim'], df['keywords_yake_claim_noun_chunk'] = zip(*results)

    # Save dataframe to json
    # df.to_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/cleantech_epo_text_data_pivot_cleaned_yake_noun_chunks.json', orient='records')
    df.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_epo_non_cleantech_text_data_pivot_claims_cleaned_yake_noun_chunks.json', orient='records')

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()