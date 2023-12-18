import re
import pandas as pd
import numpy as np
import gcld3
import multiprocessing as mp
import xml.etree.ElementTree as ET
import psycopg2
import unicodedata
from wordtrie import WordTrie
from nltk.stem import WordNetLemmatizer
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
MAX_ITER = 1
model = LogisticRegression(max_iter=1000)
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

# Database connection details
conn_patstat = psycopg2.connect(
    dbname='Patstat',
    user='tie',
    password='TIE%2023!tuhh',
    host='100.113.100.152',
    # host = '134.28.58.100',
    port=25432
)

# Database connection details for Openalex
conn_openalex = psycopg2.connect(
    dbname='openalex_db',
    user='tie',
    password='TIE%2023!tuhh',
    host='100.113.100.152',
    # host = '134.28.58.100',
    port=45432
)

def search_wrapper(trie, text):
    return [' '.join(y[0]) for y in trie.search(text, return_nodes=True)] if text else None

def clean_claim_text_xml(claim_text):
    if isinstance(claim_text, list):
        claim_text = ' '.join(claim_text)
    elif claim_text is None:
        return None
    claim_text = re.sub(r'<!--.*?-->', ' ', claim_text)
    claim_text = '<root>' + claim_text + '</root>'
    try:
        root = ET.fromstring(claim_text)
        cleaned_texts = [elem.text for elem in root.findall('.//claim-text') if elem.text]
        cleaned_texts = [re.sub(r'^\d+\.\s*', '', text) for text in cleaned_texts]
        cleaned_text = ' '.join(cleaned_texts)
    except ET.ParseError:
        return claim_text
    return cleaned_text.strip()

def clean_and_lemmatize(text):
    text = unicodedata.normalize("NFKD", text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    text = re.sub(r"[^a-zA-Z- ]", " ", text).lower().strip()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def make_wordtrie(keyword_list):
    trie = WordTrie()
    if keyword_list is None:
        return None
    for i, keyword in enumerate(keyword_list):
        if isinstance(keyword, str):
            trie.add(keyword, i)
    print(f"Added {len(keyword_list)} keywords to trie")
    return trie

def train_evaluate_model(i, model, X_train, X_test, y_train, y_test, df_cleantech, list_classification_reports):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    feature_importance = model.coef_[0]
    feature_names = Vectorizer.get_feature_names_out()
    sorted_keywords = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
    df_keywords_importance = pd.DataFrame(sorted_keywords, columns=['keyword_yake_lemma', f'logistic_regression_importance_iteration_{i}'])
    df_cleantech = pd.merge(df_cleantech, df_keywords_importance, on='keyword_yake_lemma', how='left')
    list_classification_reports.append(classification_report(y_test, predictions, output_dict=True))
    return df_cleantech

def process_text_parallel(df, column, function):
    with mp.Pool(min(mp.cpu_count(), 12)) as pool:
        results = pool.map(function, df[column])
    df[column] = results
    return df

def trie_search_parallel(df, column, trie):
    with mp.Pool(min(mp.cpu_count(), 12)) as pool:
        # Create a partial function that has the trie already assigned
        from functools import partial
        func = partial(search_wrapper, trie)
        results = pool.map(func, df[column])
    df[column] = results
    return df


# Prepare Cleantech Data
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

Vectorizer = CountVectorizer(
    vocabulary = cleantech_list,
    ngram_range = (1, 4),
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

    # Randomly sample len(publn_nr_list) patents from Patstat Postgres database that are not in publn_nr_list
    epo_non_cleantech_query = f"""
        SELECT epo_publn_nr, appln_kind, appln_text
        FROM ep_fulltext_data
        WHERE epo_publn_nr NOT IN {tuple(publn_nr_list)}
        AND appln_lng = 'en'
        AND appln_comp ='CLAIM'
        ORDER BY RANDOM()
        LIMIT {len(publn_nr_list)}
    """
    cursor = conn_patstat.cursor()
    cursor.execute(epo_non_cleantech_query)
    df_epo_non_cleantech_sample = pd.DataFrame(cursor.fetchall(), columns=['epo_publn_nr', 'appln_kind', 'appln_text'])
    cursor.close()
    df_epo_non_cleantech_sample['appln_text'] = df_epo_non_cleantech_sample['appln_text'].astype(str)
    order = {'B9': 11, 'B8': 10, 'B3': 9, 'B2': 8, 'B1': 7, 'A9': 6, 'A8': 5, 'A4': 4, 'A3': 3, 'A2': 2, 'A1': 1}
    df_epo_non_cleantech_sample.sort_values(by=['epo_publn_nr', 'appln_kind'], key=lambda x: x.map(order), ascending=False, inplace=True)
    df_epo_non_cleantech_sample.drop_duplicates(subset=['epo_publn_nr'], keep='first', inplace=True)
    df_epo_non_cleantech_sample = process_text_parallel(df_epo_non_cleantech_sample, 'appln_text', clean_claim_text_xml)
    df_epo_non_cleantech_sample = process_text_parallel(df_epo_non_cleantech_sample, 'appln_text', clean_and_lemmatize)
    print(df_epo_non_cleantech_sample.head()) # FOR TESTING
    df_epo_non_cleantech_sample = trie_search_parallel(df_epo_non_cleantech_sample, 'appln_text', cleantech_trie)
    print(df_epo_non_cleantech_sample.head()) # FOR TESTING
    df_epo_non_cleantech_sample.rename(columns={'epo_publn_nr': 'id'}, inplace=True)
    df_epo_non_cleantech_sample['id'] = 'epo-' + df_epo_non_cleantech_sample['id']
    df_epo_non_cleantech_sample = df_epo_non_cleantech_sample[['id', 'trie']]

    # Randomly sample len(patent_id_list) patents from PatentsView Postgres database that are not in patent_id_list
    uspto_non_cleantech_query = """
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
    cursor = conn_patstat.cursor()
    cursor.execute(uspto_non_cleantech_query)
    df_uspto_non_cleantech_sample = pd.DataFrame(cursor.fetchall(), columns=['patent_id', 'claim_text', 'claim_sequence'])
    cursor.close()
    df_uspto_non_cleantech_sample.sort_values(by=['patent_id', 'claim_sequence'], inplace=True)
    df_uspto_non_cleantech_sample['claim_text'] = df_uspto_non_cleantech_sample['claim_text'].str.replace(r'^\d+\.\s', ' ', regex=True)
    df_uspto_non_cleantech_sample = df_uspto_non_cleantech_sample.groupby('patent_id')['claim_text'].apply(' '.join).reset_index()
    df_uspto_non_cleantech_sample = process_text_parallel(df_uspto_non_cleantech_sample, 'claim_text', clean_and_lemmatize)
    df_uspto_non_cleantech_sample = trie_search_parallel(df_uspto_non_cleantech_sample, 'claim_text', cleantech_trie)
    df_uspto_non_cleantech_sample.rename(columns={'patent_id': 'id'}, inplace=True)
    df_uspto_non_cleantech_sample['id'] = 'us-' + df_uspto_non_cleantech_sample['id']
    df_uspto_non_cleantech_sample = df_uspto_non_cleantech_sample[['id', 'trie']]

    # Randomly sample len(oaid_list) patents from OpenALEX Postgres database that are not in oaid_list
    rel_non_cleantech_query = f"""
        SELECT id, abstract_inverted_index
        FROM openalex.works
        WHERE id NOT IN {tuple(oaid_list)}
        AND abstract_inverted_index IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {len(oaid_list)}
    """
    cursor = conn_openalex.cursor()
    cursor.execute(rel_non_cleantech_query)
    df_rel_non_cleantech_sample = pd.DataFrame(cursor.fetchall(), columns=['id', 'abstract_inverted_index'])
    cursor.close()
    df_rel_non_cleantech_sample = process_text_parallel(df_rel_non_cleantech_sample, 'abstract', clean_and_lemmatize)
    df_rel_non_cleantech_sample = trie_search_parallel(df_rel_non_cleantech_sample, 'abstract', cleantech_trie)
    df_rel_non_cleantech_sample['id'] = 'rel-' + df_rel_non_cleantech_sample['id']
    df_rel_non_cleantech_sample = df_rel_non_cleantech_sample[['id', 'trie']]
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
            word_index.sort(key=lambda x: x[1])
            abstract = ' '.join([i[0] for i in word_index])
            df_rel_non_cleantech_sample.at[index, 'abstract'] = abstract
        except AttributeError:
            continue
    df_rel_non_cleantech_sample['abstract'] = df_rel_non_cleantech_sample['abstract'].astype(str)
    df_rel_non_cleantech_sample['abstract_language'] = df_rel_non_cleantech_sample['abstract'].apply(lambda x: detector.FindLanguage(text=x).language)
    df_rel_non_cleantech_sample = df_rel_non_cleantech_sample[df_rel_non_cleantech_sample['abstract_language'].str.contains('en')]

    # Concatenate samples and perform Logistic Regression
    df_epo_non_cleantech_sample['trie'] = df_epo_non_cleantech_sample['trie'].apply(lambda x: ' '.join(x))
    df_uspto_non_cleantech_sample['trie'] = df_uspto_non_cleantech_sample['trie'].apply(lambda x: ' '.join(x))
    df_rel_non_cleantech_sample['trie'] = df_rel_non_cleantech_sample['trie'].apply(lambda x: ' '.join(x))
    g_non_cleantech = pd.concat([df_epo_non_cleantech_sample, df_uspto_non_cleantech_sample, df_rel_non_cleantech_sample], ignore_index=True)
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

conn_openalex.close()
conn_patstat.close()