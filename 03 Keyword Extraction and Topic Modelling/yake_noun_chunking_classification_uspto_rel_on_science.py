import os
import pandas as pd
import yake
import spacy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Load Spacy model
nlp = spacy.load("en_core_web_lg")

# Function to initialize YAKE model
def initialize_yake():
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.25
    deduplication_algo = "seqm"
    windowSize = 5
    numOfKeywords = 25

    return yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                 dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                 features=None)

# Function to extract keywords and filter with noun chunks
def extract_and_filter(row):
    try:
        kw_extractor = initialize_yake()

        # Extract keywords with YAKE
        unfiltered_keywords = kw_extractor.extract_keywords(row['abstract'])

        # Process text with Spacy
        doc = nlp(row['abstract'])

        # Get noun chunks as a set
        noun_chunks = {chunk.text.strip().lower() for chunk in doc.noun_chunks}

        # Filter keywords based on whether they are contained within any of the noun_chunks
        filtered_keywords = [(keyword, score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]

        return unfiltered_keywords, filtered_keywords
    except:
        return '', ''


def main():
    # Import test data
    df = pd.read_json('/mnt/hdd01/patentsview/Reliance on Science - Cleantech Patents/df_oaid_Cleantech_Y02_individual_works.json')

    # Delete all rows except 'id', 'doi', 'title', 'abstract', 'oaid', 'patent_id', 'full_oaid'
    df = df[['id', 'doi', 'title', 'abstract', 'oaid', 'patent_id', 'full_oaid']]

    # Cast column 'abstract' to string
    df['abstract'] = df['abstract'].astype(str)

    # Set up multiprocessing
    num_cores = min(6, cpu_count())
    pool = Pool(num_cores)

    # Apply the function in parallel
    results = list(tqdm(pool.imap(extract_and_filter, [row for _, row in df.iterrows()]), total=len(df)))

    # Split results into separate columns
    df['keywords_yake_claim'], df['keywords_yake_claim_noun_chunk'] = zip(*results)

    # Save dataframe to json
    df.to_json('/mnt/hdd01/patentsview/Reliance on Science - Cleantech Patents/df_oaid_Cleantech_y02_individual_works_yake_noun_chunks.json', orient='records')

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()