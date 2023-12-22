import re
import pandas as pd
import numpy as np
import gcld3
import multiprocessing as mp
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sqlalchemy import create_engine, URL
tqdm.pandas()
import unicodedata
from wordtrie import WordTrie
from nltk.stem import WordNetLemmatizer
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()

MAX_ITER = 1 # Number of resampling from databases/Iterations
model = LogisticRegression(max_iter=1000) # Initialize SciKit-Learn Logistic Regression model

# Initialize language detector
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

# Database connection details
url_object_patstat = URL.create(
    drivername='postgresql+psycopg2',
    username='tie',
    password='TIE%2023!tuhh',
    # host='134.28.58.100',
    host='100.113.100.152',
    port=25432,
    database='Patstat',
)

url_object_openalex = URL.create(
    drivername='postgresql+psycopg2',
    username='tie',
    password='TIE%2023!tuhh',
    # host='134.28.58.100',
    host='100.113.100.152',
    port=45432,
    database='openalex_db',
)

engine_patstat = create_engine(url_object_patstat)
engine_openalex = create_engine(url_object_openalex)
engine_patstat.connect()
engine_openalex.connect()

def clean_claim_text_xml(claim_text):
    '''
    Extracts the text from <claim-text> tags and removes claim numbers from the beginning of the text.
    :param claim_text: A string containing the claim text in XML format.
    :return: A string containing the cleaned claim text.
    '''
    # If the input is a list, join it into a string
    if isinstance(claim_text, list):
        claim_text = ' '.join(claim_text)
    elif claim_text is None:
        return None
    # Remove all instances of <!--(.*?)-->
    claim_text = re.sub(r'<!--.*?-->', ' ', claim_text)
    # Surround the claim_text with a root tag to make it a valid XML
    claim_text = '<root>' + claim_text + '</root>'
    # Remove HTML tags with three or less characters between < and >
    # claim_text = re.sub(r'<.{0,5}>', ' ', claim_text)
    try:
        # Parse the claim_text as XML
        root = ET.fromstring(claim_text)
        # Extract all text from <claim-text> tags
        cleaned_texts = [elem.text for elem in root.findall('.//claim-text') if elem.text]
        # Remove claim numbers if they exist
        cleaned_texts = [re.sub(r'^\d+\.\s*', '', text) for text in cleaned_texts]
        # Join the cleaned texts
        cleaned_text = ' '.join(cleaned_texts)
    except ET.ParseError:
        return claim_text  # Return the original text if parsing fails
    return cleaned_text.strip()

def clean_and_lemmatize(text):
    """
    Normalize, clean, and lemmatize the input text.
    :param text: A string containing the text to be processed.
    :return: A string representing the processed text.
    """
    # Normalize the text with unicodedata
    text = unicodedata.normalize("NFKD", text).encode('ASCII', 'ignore').decode('utf-8')
    # Remove URLs, brackets, and non-alphabetic characters; convert to lowercase
    text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9- .,;!?]", "", text)
    # Lemmatize each word
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return lemmatized_text

# Build WordTrie
def make_wordtrie(keyword_list):
    '''
    Builds a WordTrie from a list of keywords.
    :param keyword_list: A list of keywords.
    :return: A WordTrie object.
    '''
    trie = WordTrie()
    if keyword_list is None:
        return None
    i = 0
    for keyword in keyword_list:
        if isinstance(keyword, str):
            trie.add(keyword, i)
            i += 1
    print(f"Added {i} keywords to trie")
    return trie

# Function to train and evaluate a given model
def train_evaluate_model(i, model, X_train, X_test, y_train, y_test, df_cleantech, list_classification_reports):
    '''
    Trains and evaluates a logistic regression model.
    :return: A DataFrame containing the keywords and their importance.
    '''
    # Train the model
    model.fit(X_train, y_train)
    # Evaluate the model
    predictions = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    feature_importance = model.coef_[0]
    feature_names = Vectorizer.get_feature_names_out()
    keywords_importance = zip(feature_names, feature_importance)
    sorted_keywords = sorted(keywords_importance, key=lambda x: x[1], reverse=True)
    df_keywords_importance = pd.DataFrame(sorted_keywords, columns=['keyword_yake_lemma', f'logistic_regression_importance_iteration_{i}'])
    df_cleantech = pd.merge(df_cleantech, df_keywords_importance, on='keyword_yake_lemma', how='left')
    # Append to classification report list
    list_classification_reports = list_classification_reports.append(classification_report(y_test, predictions, output_dict=True))
    
    return df_cleantech


##### Prepare Cleantech Data #####
co_occurrence_file = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Co-Occurrence Analysis/co_occurrence_matrix_yake_keywords_cleantech_uspto_epo_rel_ids_semantic_similarity_02.csv'
similarity_file = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Similarity Search/df_keyword_titles_cosine_similarity_radius_025_neighbors_100_noun_chunks.json'
co_occurrence_threshold = 0.01

df_cleantech_cooccurrence = pd.read_csv(co_occurrence_file, index_col=0)
df_cleantech_cooccurrence.dropna(how='all', inplace=True)
mask = df_cleantech_cooccurrence >= co_occurrence_threshold
filtered_co_occurrence_df = df_cleantech_cooccurrence[mask]
co_occurrence_list = filtered_co_occurrence_df.columns[filtered_co_occurrence_df.any()].tolist()

df_cleantech_similarity = pd.read_json(similarity_file)
similarity_series = pd.concat([df_cleantech_similarity['keyword_yake_lemma'], df_cleantech_similarity['keywords_keyword_yake_bertforpatents_embedding'].explode()], ignore_index=True)
similarity_list = similarity_series.drop_duplicates().tolist()

cleantech_list = list(set(co_occurrence_list + similarity_list))
cleantech_list = [str(keyword) for keyword in cleantech_list]

##### Initialize Vectorizer #####
Vectorizer = CountVectorizer(
    vocabulary = cleantech_list,
    ngram_range = (1, 4),
    # max_df = 0.5,
    # min_df = 0.01,
    stop_words='english',
    lowercase=True,
)
df_cleantech = pd.DataFrame(cleantech_list, columns=['keyword_yake_lemma'])
cleantech_trie = make_wordtrie(df_cleantech['keyword_yake_lemma'].tolist())

# Read Trie Searched Cleantech Data
g_epo_cleantech = pd.read_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/g_epo_cleantech_trie.csv')
g_epo_cleantech.rename(columns={'publn_nr': 'id'}, inplace=True)
g_epo_cleantech['id'] = 'epo-' + g_epo_cleantech['id'].astype(str)

g_uspto_cleantech = pd.read_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/g_uspto_cleantech_trie.csv')
g_uspto_cleantech.rename(columns={'patent_id': 'id'}, inplace=True)

g_rel_cleantech = pd.read_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/g_rel_cleantech_trie.csv')
g_rel_cleantech.rename(columns={'oaid': 'id'}, inplace=True)
g_epo_cleantech.dropna(subset=['trie'], inplace=True)
g_uspto_cleantech.dropna(subset=['trie'], inplace=True)
g_rel_cleantech.dropna(subset=['trie'], inplace=True)
g_epo_cleantech['trie'] = g_epo_cleantech['trie'].apply(lambda x: ' '.join(eval(x)))
g_uspto_cleantech['trie'] = g_uspto_cleantech['trie'].apply(lambda x: ' '.join(eval(x)))
g_rel_cleantech['trie'] = g_rel_cleantech['trie'].apply(lambda x: ' '.join(eval(x)))

g_cleantech = pd.concat([g_epo_cleantech, g_uspto_cleantech, g_rel_cleantech], ignore_index=True)
g_cleantech = g_cleantech[['id', 'trie']]
g_cleantech_matrix = Vectorizer.transform(g_cleantech['trie'])

publn_nr_list = g_epo_cleantech['id'].tolist()
patent_id_list = g_uspto_cleantech['id'].tolist()
oaid_list = g_rel_cleantech['id'].tolist()
publn_nr_list = [publn_nr[4:] for publn_nr in publn_nr_list]
patent_id_list = [patent_id[3:] for patent_id in patent_id_list]
oaid_list = [oaid[4:] for oaid in oaid_list]
oaid_list = ['https://openalex.org/W' + str(oaid) for oaid in oaid_list]

list_classification_reports = []

for i in range(MAX_ITER):
    print(f"Starting iteration {i+1} of {MAX_ITER}...")

    ##### Randomly sample len(publn_nr_list) patents from Patstat Postgres database that are not in publn_nr_list #####
    epo_non_cleantech_query = f"""
        SELECT epo_publn_nr, appln_kind, appln_text
        FROM ep_fulltext_data
        WHERE epo_publn_nr NOT IN {tuple(publn_nr_list)}
        AND appln_lng = 'en'
        AND appln_comp ='CLAIM'
        ORDER BY RANDOM()
        LIMIT {len(publn_nr_list)}
    """
    df_epo_non_cleantech_sample = pd.read_sql(epo_non_cleantech_query, engine_patstat)
    # Cast column appln_text to string
    df_epo_non_cleantech_sample['appln_text'] = df_epo_non_cleantech_sample['appln_text'].astype(str)
    # Create a custom order for appln_kind
    order = {'B9': 11, 'B8': 10, 'B3': 9, 'B2': 8, 'B1': 7, 'A9': 6, 'A8': 5, 'A4': 4, 'A3': 3, 'A2': 2, 'A1': 1}
    # Sort df_epo_non_cleantech_sample by appln_kind, keep only highest appln_kind per epo_publn_nr
    df_epo_non_cleantech_sample.sort_values(by=['epo_publn_nr', 'appln_kind'], key=lambda x: x.map(order), ascending=False, inplace=True)
    df_epo_non_cleantech_sample.drop_duplicates(subset=['epo_publn_nr'], keep='first', inplace=True)
    # Apply clean_claim_text_xml function to appln_text column
    df_epo_non_cleantech_sample['appln_text'] = df_epo_non_cleantech_sample['appln_text'].apply(clean_claim_text_xml)
    # Create a pool of workers
    pool = mp.Pool(min(mp.cpu_count(),6))
    # Apply the function to the 'cleaned_claims' column using the pool of workers
    results = []
    for result in pool.imap(clean_and_lemmatize, df_epo_non_cleantech_sample['appln_text']):
        results.append(result)
    df_epo_non_cleantech_sample['appln_text'] = results
    pool.close()
    # Perform trie search
    df_epo_non_cleantech_sample['trie'] = df_epo_non_cleantech_sample['appln_text'].apply(lambda x: cleantech_trie.search(x, return_nodes=True))
    df_epo_non_cleantech_sample['trie'] = df_epo_non_cleantech_sample['trie'].apply(lambda x: [' '.join(y[0]) for y in x] if len(x) > 0 else None)
    # Delete all rows where trie is NaN or empty
    df_epo_non_cleantech_sample.dropna(subset=['trie'], inplace=True)
    df_epo_non_cleantech_sample = df_epo_non_cleantech_sample[['epo_publn_nr', 'trie']]
    # Rename 'epo_publn_nr' to 'id' and append 'epo-' and rename 'appln_text' to 'text'
    df_epo_non_cleantech_sample.rename(columns={'epo_publn_nr': 'id'}, inplace=True)
    df_epo_non_cleantech_sample['id'] = 'epo-' + df_epo_non_cleantech_sample['id']
    # Drop all other columns
    df_epo_non_cleantech_sample = df_epo_non_cleantech_sample[['id', 'trie']]


    ##### Randomly sample len(patent_id_list) patents from PatentsView Postgres database that are not in patent_id_list #####
    uspto_non_cleantech_query = f"""
        WITH distinct_ids AS (
            SELECT DISTINCT patent_id
            FROM public.us_claims
            WHERE patent_id NOT IN {tuple(patent_id_list)}
        ),
        ordered_ids AS (
            SELECT patent_id
            FROM distinct_ids
            ORDER BY RANDOM()
            LIMIT {len(patent_id_list)}
        )
        SELECT us_claims.*
        FROM public.us_claims
        JOIN ordered_ids
        ON us_claims.patent_id = ordered_ids.patent_id
    """
    df_uspto_non_cleantech_sample = pd.read_sql(uspto_non_cleantech_query, engine_patstat)
    # Sort df_uspto_non_cleantech_sample by patent_id and then by claim_sequence
    df_uspto_non_cleantech_sample.sort_values(by=['patent_id', 'claim_sequence'], inplace=True)
    # Remove the claim number from the beginning of the claim text
    df_uspto_non_cleantech_sample['claim_text'] = df_uspto_non_cleantech_sample['claim_text'].str.replace(r'^\d+\.\s', ' ', regex=True)
    # Group by patent_id and concatenate all claims into a single string
    df_uspto_non_cleantech_sample = df_uspto_non_cleantech_sample.groupby('patent_id')['claim_text'].apply(' '.join).reset_index()
    # Create a pool of workers
    pool = mp.Pool(min(mp.cpu_count(),6))
    # Apply the function to the 'claim_fulltext' column using the pool of workers
    results = []
    for result in pool.imap(clean_and_lemmatize, df_uspto_non_cleantech_sample['claim_text']):
        results.append(result)
    df_uspto_non_cleantech_sample['claim_text'] = results
    pool.close()
    # Cast column claim_text to string
    df_uspto_non_cleantech_sample['claim_text'] = df_uspto_non_cleantech_sample['claim_text'].astype(str)
    # Perform trie search
    df_uspto_non_cleantech_sample['trie'] = df_uspto_non_cleantech_sample['claim_text'].apply(lambda x: cleantech_trie.search(x, return_nodes=True))
    df_uspto_non_cleantech_sample['trie'] = df_uspto_non_cleantech_sample['trie'].apply(lambda x: [' '.join(y[0]) for y in x] if len(x) > 0 else None)
    # Delete all rows where trie is NaN or empty
    df_uspto_non_cleantech_sample.dropna(subset=['trie'], inplace=True)
    df_uspto_non_cleantech_sample = df_uspto_non_cleantech_sample[['patent_id', 'trie']]
    # Rename 'patent_id' to 'id' and append 'us-' and rename 'claim_text' to 'text'
    df_uspto_non_cleantech_sample.rename(columns={'patent_id': 'id'}, inplace=True)
    df_uspto_non_cleantech_sample['id'] = 'us-' + df_uspto_non_cleantech_sample['id']
    # Drop all other columns
    df_uspto_non_cleantech_sample = df_uspto_non_cleantech_sample[['id', 'trie']]


    ##### Randomly sample len(oaid_list) patents from OpenALEX Postgres database that are not in oaid_list #####
    rel_non_cleantech_query = f"""
        SELECT id, abstract_inverted_index
        FROM openalex.works
        WHERE id NOT IN {tuple(oaid_list)}
        AND abstract_inverted_index IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {len(oaid_list)}
    """
    df_rel_non_cleantech_sample = pd.read_sql(rel_non_cleantech_query, engine_openalex)
    # Delete all empty rows in column abstract_inverted_index with value None
    df_rel_non_cleantech_sample.dropna(subset=['abstract_inverted_index'], inplace=True)
    # Iterate over abstract_inverted_index columnc
    for index, row in df_rel_non_cleantech_sample.iterrows():
        word_index = []
        try:
            for key, value in row['abstract_inverted_index'].items():
                if key == 'InvertedIndex':
                    for innerkey, innervalue in value.items():
                        for innerindex in innervalue:
                            word_index.append([innerkey, innerindex])
            # Sort list by index
            word_index.sort(key=lambda x: x[1])
            # Join first element of each list in word_index
            abstract = ' '.join([i[0] for i in word_index])
            # Add column abstract to result dataframe
            df_rel_non_cleantech_sample.at[index, 'abstract'] = abstract
        except AttributeError:
            continue
    # Cast column abstract to string
    df_rel_non_cleantech_sample['abstract'] = df_rel_non_cleantech_sample['abstract'].astype(str)
    # Delete all rows where abstract language is not English
    df_rel_non_cleantech_sample['abstract_language'] = df_rel_non_cleantech_sample['abstract'].apply(lambda x: detector.FindLanguage(text=x).language)
    df_rel_non_cleantech_sample = df_rel_non_cleantech_sample[df_rel_non_cleantech_sample['abstract_language'].str.contains('en')]
    # Create a pool of workers
    pool = mp.Pool(min(mp.cpu_count(),6))
    # Apply the function to the 'abstract' column using the pool of workers
    results = []
    for result in pool.imap(clean_and_lemmatize, df_rel_non_cleantech_sample['abstract']):
        results.append(result)
    df_rel_non_cleantech_sample['abstract'] = results
    pool.close()
    # Perform trie search
    df_rel_non_cleantech_sample['trie'] = df_rel_non_cleantech_sample['abstract'].apply(lambda x: cleantech_trie.search(x, return_nodes=True))
    df_rel_non_cleantech_sample['trie'] = df_rel_non_cleantech_sample['trie'].apply(lambda x: [' '.join(y[0]) for y in x] if len(x) > 0 else None)
    # Delete all rows where trie is NaN or empty
    df_rel_non_cleantech_sample.dropna(subset=['trie'], inplace=True)
    df_rel_non_cleantech_sample = df_rel_non_cleantech_sample[['id', 'trie']]
    # Append 'rel-' to id
    df_rel_non_cleantech_sample['id'] = 'rel-' + df_rel_non_cleantech_sample['id']


    ##### Concatenate samples and perform Logistic Regression #####
    # Concatenate the list of strings in trie column to a single string
    df_epo_non_cleantech_sample['trie'] = df_epo_non_cleantech_sample['trie'].apply(lambda x: ' '.join(x))
    df_uspto_non_cleantech_sample['trie'] = df_uspto_non_cleantech_sample['trie'].apply(lambda x: ' '.join(x))
    df_rel_non_cleantech_sample['trie'] = df_rel_non_cleantech_sample['trie'].apply(lambda x: ' '.join(x))
    # Concatenate the three DataFrames
    g_non_cleantech = pd.concat([df_epo_non_cleantech_sample, df_uspto_non_cleantech_sample, df_rel_non_cleantech_sample], ignore_index=True)

    # Build vectorized Matrices
    g_non_cleantech_matrix = Vectorizer.transform(g_non_cleantech['trie'])
    
    # Concatenate data for train_test_split
    X = vstack([g_cleantech_matrix, g_non_cleantech_matrix])
    y = np.concatenate([np.ones(g_cleantech_matrix.shape[0]), np.zeros(g_non_cleantech_matrix.shape[0])])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and evaluate the model
    df_cleantech = train_evaluate_model(i, model, X_train, X_test, y_train, y_test, df_cleantech, list_classification_reports)

    print(df_cleantech.head())

    print(list_classification_reports)