import pandas as pd
import re
import yake
import spacy
import unicodedata
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gcld3

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

# Function to extract keywords and filter with noun chunks for both title and abstract
def extract_and_filter(row):
    results = {'keywords_yake_title': [], 'keywords_yake_title_noun_chunk': [], 'keywords_yake_abstract': [], 'keywords_yake_abstract_noun_chunk': []}
    for column in ['title', 'abstract']:
        text_content = row[column]

        # Initialize results for each column to handle non-English or missing content
        results[f'keywords_yake_{column}'] = []
        results[f'keywords_yake_{column}_noun_chunk'] = []

        if pd.isna(text_content) or detector.FindLanguage(text_content).language != 'en':
            print(f"Skipping or non-English content found for column '{column}' in oaid number: {row['oaid']}")
            continue
        
        # Normalize the text
        text_content = unicodedata.normalize("NFKD", text_content).encode('ASCII', 'ignore').decode('utf-8')
        text_content = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text_content)
        text_content = re.sub(r"https?:\/\/\S+", "", text_content)
        text_content = re.sub(r"[^a-zA-Z0-9- .,;!?]", "", text_content)
        
        # Extract keywords using YAKE
        unfiltered_keywords = yake_extractor.extract_keywords(text_content)
        doc = nlp(text_content)
        noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        filtered_keywords = [(keyword.lower(), score) for keyword, score in unfiltered_keywords if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
        filtered_keywords = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in filtered_keywords]
        filtered_keywords = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in filtered_keywords]
        
        if column == 'title':
            results['keywords_yake_title'] = unfiltered_keywords
            results['keywords_yake_title_noun_chunk'] = filtered_keywords
        else:  # column == 'abstract'
            results['keywords_yake_abstract'] = unfiltered_keywords
            results['keywords_yake_abstract_noun_chunk'] = filtered_keywords

    return results

def main():
    # Import test data
    df = pd.read_csv('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_oaids_non_cleantech.csv', index_col=0)

    # Keep only necessary columns
    df = df[['doi', 'title', 'abstract', 'oaid']]
    df['title'] = df['title'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['abstract'] = df['abstract'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['title'] = df['title'].astype(str)
    df['abstract'] = df['abstract'].astype(str)

    print(f"Starting keyword extraction and filtering for {len(df)} records")

    # Set up multiprocessing
    num_cores = min(6, cpu_count())
    pool = Pool(num_cores)

    # Apply the function in parallel
    results = list(tqdm(pool.imap(extract_and_filter, [row for _, row in df.iterrows()]), total=len(df)))

    # Lists to hold the results for each keyword field
    keywords_yake_title = []
    keywords_yake_title_noun_chunk = []
    keywords_yake_abstract = []
    keywords_yake_abstract_noun_chunk = []

    # Loop through results and build the lists
    for result in results:
        keywords_yake_title.append('; '.join([kw[0] for kw in result['keywords_yake_title']]))
        keywords_yake_title_noun_chunk.append('; '.join([kw[0] for kw in result['keywords_yake_title_noun_chunk']]))
        keywords_yake_abstract.append('; '.join([kw[0] for kw in result['keywords_yake_abstract']]))
        keywords_yake_abstract_noun_chunk.append('; '.join([kw[0] for kw in result['keywords_yake_abstract_noun_chunk']]))

    # Assign the lists to the DataFrame, ensuring each row gets its corresponding keywords
    df['keywords_yake_title'] = keywords_yake_title
    df['keywords_yake_title_noun_chunk'] = keywords_yake_title_noun_chunk
    df['keywords_yake_abstract'] = keywords_yake_abstract
    df['keywords_yake_abstract_noun_chunk'] = keywords_yake_abstract_noun_chunk

    # Save dataframe to json
    df.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_oaids_non_cleantech_title_abstract_lang_detect_yake_noun_chunks.json', orient='records')

    print(f"Finished processing {len(df)} records. Saved to json.")

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
