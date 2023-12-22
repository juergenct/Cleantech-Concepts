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
        if row['claim_fulltext'] is None:
            print("No claims found for patent number: " + str(row['patent_id']))
            return [], []
        
        # Normalize the text with unicodedata
        row['claim_fulltext'] = unicodedata.normalize("NFKD", row['claim_fulltext']).encode('ASCII', 'ignore').decode('utf-8')
        # row['claim_fulltext'] = re.sub(r"[^a-zA-Z- ]|^https?:\/\/.*[\r\n]*|\[.*?\]|\(.*?\)|\{.*?\}", " ", row['claim_fulltext']).lower().strip()
        row['claim_fulltext'] = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", row['claim_fulltext'])
        row['claim_fulltext'] = re.sub(r"https?:\/\/\S+", "", row['claim_fulltext'])
        row['claim_fulltext'] = re.sub(r"[^a-zA-Z0-9- .,;!?]", "", row['claim_fulltext'])
        unfiltered_keywords = yake_extractor.extract_keywords(row['claim_fulltext']) # Extract keywords using YAKE
        doc = nlp(row['claim_fulltext'])
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        return unfiltered_keywords, filtered_keywords
    except:
        print("Error in patent number: " + str(row['patent_id']))
        return [], []

def main():
    # Import test data
    df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_fulltext_cleantech.json')
    # df = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/g_uspto_non_cleantech_claims_fulltext.json')

    # Set up multiprocessing
    num_cores = min(12, cpu_count())
    pool = Pool(num_cores)

    # Cast column 'claim_fulltext' to string
    df['claim_fulltext'] = df['claim_fulltext'].astype(str)

    # Apply the function in parallel
    results = list(tqdm(pool.imap(extract_and_filter, [row for _, row in df.iterrows()]), total=len(df)))

    # Split results into separate columns
    df['keywords_yake_claim'], df['keywords_yake_claim_noun_chunk'] = zip(*results)

    # Save dataframe to json
    df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_cleantech_yake_noun_chunks.json', orient='records')
    # df.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/g_uspto_non_cleantech_claims_fulltext_yake_noun_chunks.json', orient='records')

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()