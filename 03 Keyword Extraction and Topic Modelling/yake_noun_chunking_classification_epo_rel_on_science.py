import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import gcld3
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

# Initialize language detector
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

# Function to extract keywords and filter with noun chunks
def extract_and_filter(row):
    try:
        if row['abstract'] is None:
            print("No claims found for oaid number: " + str(row['oaid']))
            return [], []
        if detector.FindLanguage(row['abstract']).language != 'en':
            print("Non-English abstract found for oaid number: " + str(row['oaid']))
            return [], []
        
        # Normalize the text with unicodedata
        row['abstract'] = unicodedata.normalize("NFKD", row['abstract']).encode('ASCII', 'ignore').decode('utf-8')
        # row['abstract'] = re.sub(r"[^a-zA-Z- ]|^https?:\/\/.*[\r\n]*|\[.*?\]|\(.*?\)|\{.*?\}", " ", row['abstract']).lower().strip()
        row['abstract'] = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", row['abstract'])
        row['abstract'] = re.sub(r"https?:\/\/\S+", "", row['abstract'])
        row['abstract'] = re.sub(r"[^a-zA-Z- ]", " ", row['abstract']).lower().strip()
        unfiltered_keywords = yake_extractor.extract_keywords(row['abstract']) # Extract keywords using YAKE
        doc = nlp(row['abstract'])
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        return unfiltered_keywords, filtered_keywords
    except:
        print("Error in oaid number: " + str(row['oaid']))
        return [], []

def main():
    # Import test data
    df = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/Reliance on Science/cleantech_epo_rel_on_science_abstract.json')

    # Delete all rows except 'id', 'doi', 'title', 'abstract', 'oaid', 'oaid', 'full_oaid'
    df = df[['id', 'doi', 'title', 'abstract', 'oaid', 'full_oaid']]

    # Cast column 'abstract' to string
    df['abstract'] = df['abstract'].astype(str)

    # Set up multiprocessing
    num_cores = min(2, cpu_count())
    pool = Pool(num_cores)

    # Apply the function in parallel
    results = list(tqdm(pool.imap(extract_and_filter, [row for _, row in df.iterrows()]), total=len(df)))

    # Split results into separate columns
    df['keywords_yake_claim'], df['keywords_yake_claim_noun_chunk'] = zip(*results)

    # Save dataframe to json
    df.to_json('/mnt/hdd01/PATSTAT Working Directory/Reliance on Science/cleantech_epo_rel_on_science_abstract_lang_detect_yake_noun_chunks.json', orient='records')

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()