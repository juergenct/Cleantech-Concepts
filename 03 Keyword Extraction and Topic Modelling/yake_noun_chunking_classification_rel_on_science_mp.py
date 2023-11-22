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
        unfiltered_keywords = kw_extractor.extract_keywords(row['abstract'])
        doc = nlp(row['abstract'])
        noun_chunks = {chunk.text.strip().lower() for chunk in doc.noun_chunks}
        filtered_keywords = [(keyword, score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        return unfiltered_keywords, filtered_keywords
    except:
        return '', ''

def process_chunk(chunk):
    num_cores = min(12, cpu_count())
    with Pool(num_cores) as pool:
        results = pool.map(extract_and_filter, [row for _, row in chunk.iterrows()])
    return results

def main():
    chunk_size = 10000  # Adjust chunk size based on your system's capabilities
    total_rows = sum(1 for _ in open('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_oaids_non_cleantech.csv', 'r')) - 1  # Adjust file path
    reader = pd.read_csv('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_oaids_non_cleantech.csv', index_col=0, chunksize=chunk_size)

    processed_data = []
    with tqdm(total=total_rows) as pbar:
        for chunk in reader:
            chunk['abstract'] = chunk['abstract'].astype(str)
            chunk_results = process_chunk(chunk)
            processed_data.extend(chunk_results)
            pbar.update(chunk_size)

    # Convert processed data to DataFrame and save
    df = pd.DataFrame(processed_data, columns=['keywords_yake_claim', 'keywords_yake_claim_noun_chunk'])
    df.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_oaids_non_cleantech_yake_noun_chunks.json', orient='records')

if __name__ == '__main__':
    main()
