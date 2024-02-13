import pandas as pd
import numpy as np
import re
import yake
import spacy
import unicodedata
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

cleantech = 1

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
def extract_and_filter(row, column_name):
    text_content = row[column_name]
    if pd.isna(text_content):
        print(f"No content found for column '{column_name}' in patent number: {row['patent_id']}")
        return [], []
    
    try:
        # Normalize the text
        text_content = unicodedata.normalize("NFKD", text_content).encode('ASCII', 'ignore').decode('utf-8')
        text_content = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text_content)
        text_content = re.sub(r"https?:\/\/\S+", "", text_content)
        text_content = re.sub(r"[^a-zA-Z0-9- .,;!?]", "", text_content)
        
        unfiltered_keywords = yake_extractor.extract_keywords(text_content) # Extract keywords using YAKE
        doc = nlp(text_content)
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        return unfiltered_keywords, filtered_keywords
    except Exception as e:
        print(f"Error in processing column '{column_name}' for patent number: {row['patent_id']} - {e}")
        return [], []

def process_columns(row):
    columns_to_process = ['patent_title', 'patent_abstract', 'claim_fulltext'] # Add or remove column names as needed
    results = {}
    for column in columns_to_process:
        unfiltered_keywords, filtered_keywords = extract_and_filter(row, column)
        results[f'keywords_yake_{column}'] = unfiltered_keywords
        results[f'keywords_yake_{column}_noun_chunk'] = filtered_keywords
    return results

def main():
    # Cleantech
    if cleantech == 1:
        df = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_fulltext_cleantech.json')
        df_abstract = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/df_patentsview_patent_cpc_grouped_cleantech.json')
        df = pd.merge(df, df_abstract, on='patent_id', how='left')
        df = df[['patent_id', 'patent_title', 'patent_abstract', 'claim_fulltext', 'cpc']]
        df['patent_title'] = df['patent_title'].apply(lambda title: ' '.join(title) if isinstance(title, list) else title)
        df['patent_abstract'] = df['patent_abstract'].apply(lambda abstract: ' '.join(abstract) if isinstance(abstract, list) else abstract)
        df['claim_fulltext'] = df['claim_fulltext'].apply(lambda claim: ' '.join(claim) if isinstance(claim, list) else claim)
        df['patent_title'] = df['patent_title'].astype(str)
        df['patent_abstract'] = df['patent_abstract'].astype(str)
        df['claim_fulltext'] = df['claim_fulltext'].astype(str)
    
    # Non-cleantech
    elif cleantech == 0:
        df = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/g_uspto_non_cleantech_claims_fulltext.json')
        df_abstract = pd.read_csv('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/g_uspto_non_cleantech_abstract.csv')
        df = pd.merge(df, df_abstract, on='patent_id', how='left')
        df = df[['patent_id', 'patent_title', 'patent_abstract', 'claim_fulltext']]
        df['patent_title'] = df['patent_title'].apply(lambda title: ' '.join(title) if isinstance(title, list) else title)
        df['patent_abstract'] = df['patent_abstract'].apply(lambda abstract: ' '.join(abstract) if isinstance(abstract, list) else abstract)
        df['claim_fulltext'] = df['claim_fulltext'].apply(lambda claim: ' '.join(claim) if isinstance(claim, list) else claim)
        df['patent_title'] = df['patent_title'].astype(str)
        df['patent_abstract'] = df['patent_abstract'].astype(str)
        df['claim_fulltext'] = df['claim_fulltext'].astype(str)
    print(f"Starting to process {len(df)} {'cleantech' if cleantech == 1 else 'non-cleantech'} patents...")
    
    # Prepare the data for multiprocessing
    rows = [row for _, row in df.iterrows()]

    # Use multiprocessing Pool efficiently
    with Pool(min(10, cpu_count())) as pool:
        # imap returns an iterator that we can directly consume
        results = list(tqdm(pool.imap(process_columns, rows), total=len(df)))

    # Ensure results are directly applicable back to the DataFrame
    for i, result in enumerate(results):
        for key, value in result.items():
            if key not in df:
                df[key] = np.nan
            # Process and assign the results as before
            df[key] = df[key].astype(str)
            df.at[i, key] = '; '.join([kw[0] for kw in value])

    # Save or further process df
    print(f"Finished processing {len(df)} patents.")

    # Save dataframe to json
    if cleantech == 1:
        df.to_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_title_abstract_claims_cleantech_yake_noun_chunks.json', orient='records')
    elif cleantech == 0:
        df.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/g_uspto_non_cleantech_title_abstract_claims_fulltext_yake_noun_chunks.json', orient='records')

    pool.close()
    pool.join()

    print(f"Finished processing {len(df)} {'cleantech' if cleantech == 1 else 'non-cleantech'} patents")

if __name__ == '__main__':
    main()

