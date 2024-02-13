import pandas as pd
import numpy as np
import re
import yake
import spacy
import unicodedata
from nltk.stem import WordNetLemmatizer
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

cleantech = 1

# Function to extract keywords and filter with noun chunks for a given column
def extract_and_filter(row, column_name):
    text_content = row[column_name]
    if pd.isna(text_content):
        print(f"No content found for column '{column_name}' in publication number: {row['publn_nr']}")
        return [], []
    
    try:
        # Normalize the text
        text_content = unicodedata.normalize("NFKD", text_content).encode('ASCII', 'ignore').decode('utf-8')
        text_content = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text_content)
        text_content = re.sub(r"https?:\/\/\S+", "", text_content)
        text_content = re.sub(r"[^a-zA-Z0-9- .,;!?]", "", text_content)
        
        unfiltered_keywords = yake_extractor.extract_keywords(text_content)
        doc = nlp(text_content)
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        return unfiltered_keywords, filtered_keywords
    except Exception as e:
        print(f"Error in processing column '{column_name}' for publication number: {row['publn_nr']} - {e}")
        return [], []

def process_columns(row):
    columns_to_process = ['patent_title', 'patent_abstract', 'claim_fulltext'] # Specify the columns to process
    results = {}
    for column in columns_to_process:
        unfiltered_keywords, filtered_keywords = extract_and_filter(row, column)
        results[f'keywords_yake_{column}'] = unfiltered_keywords
        results[f'keywords_yake_{column}_noun_chunk'] = filtered_keywords
    return results

def main():
    # Import data
    if cleantech ==1:
        df = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/cleantech_epo_text_data_pivot_cleaned.json')
        df_abstract = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/cleantech_epo_text_data_pivot_cleaned_abstr.json', lines=True)
    elif cleantech == 0:
        df = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_epo_non_cleantech_text_data_pivot_claims_cleaned.json')
        df_abstract = pd.read_csv('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_epo_non_cleantech_text_data_abstr_cleaned.csv')
    
    # Merge dataframes on 'publn_nr'
    df = pd.merge(df, df_abstract[['publn_nr', 'cleaned_abstr']], on='publn_nr', how='left')
    df = df[['publn_nr', 'TITLE', 'cleaned_abstr', 'cpc_class_symbol', 'cleaned_claims']]

    df['TITLE'] = df['TITLE'].apply(lambda title: ' '.join(title) if isinstance(title, list) else title)
    df['cleaned_abstr'] = df['cleaned_abstr'].apply(lambda abstr: ' '.join(abstr) if isinstance(abstr, list) else abstr)
    df['cleaned_claims'] = df['cleaned_claims'].apply(lambda claims: ' '.join(claims) if isinstance(claims, list) else claims)

    # Ensure all text columns are string type
    df['TITLE'] = df['TITLE'].astype(str)
    df['cleaned_abstr'] = df['cleaned_abstr'].astype(str)
    df['cleaned_claims'] = df['cleaned_claims'].astype(str)
    df.rename(columns={'TITLE': 'patent_title', 'cleaned_abstr': 'patent_abstract', 'cleaned_claims': 'claim_fulltext'}, inplace=True)

    print(f"Starting to process {len(df)} patents...")

    # Prepare the data for multiprocessing
    rows = [row for _, row in df.iterrows()]

    # Use multiprocessing Pool efficiently
    with Pool(min(4, cpu_count())) as pool:
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

    print(df.columns)

    # Save dataframe to json
    if cleantech == 1:
        df.to_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/cleantech_epo_text_data_pivot_cleaned_title_abstract_claims_yake_noun_chunks.json', orient='records')
    elif cleantech == 0:
        df.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_epo_non_cleantech_text_data_pivot_title_abstract_claims_yake_noun_chunks.json', orient='records')

    pool.close()
    pool.join()

    print(f"Finished processing {len(df)} patents")

if __name__ == '__main__':
    main()