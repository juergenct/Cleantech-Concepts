import re
import pandas as pd
import numpy as np
import gcld3
import yake
import multiprocessing as mp
import xml.etree.ElementTree as ET
import psycopg2
import unicodedata
import spacy
from wordtrie import WordTrie
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
MAX_ITER = 1
model = LogisticRegression(max_iter=1000)
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
nlp = spacy.load("en_core_web_lg")
yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.25, dedupFunc="seqm", windowsSize=5, top=25, features=None)
stopwords = stopwords.words('english')

# Database connection details
conn_patstat = psycopg2.connect(
    dbname='Patstat',
    user='tie',
    password='TIE%2023!tuhh',
    # host='100.113.100.152',
    host = '134.28.58.100',
    port=25432
)

# Database connection details for Openalex
conn_openalex = psycopg2.connect(
    dbname='openalex_db',
    user='tie',
    password='TIE%2023!tuhh',
    # host='100.113.100.152',
    host = '134.28.58.100',
    port=45432
)

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

def clean_text(text):
    text = unicodedata.normalize("NFKD", text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r"\[.*?\]|\(.*?\)|\{.*?\}", "", text)
    text = re.sub(r"https?:\/\/\S+", "", text)
    text = re.sub(r"[^a-zA-Z- ]", " ", text).lower().strip()
    return ' '.join([word for word in text.split()])

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

def extract_yake_keywords(text):
    keyword_yake = yake_extractor.extract_keywords(text)
    doc = nlp(text)
    noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
    keyword_yake_noun_chunk = [(keyword.lower(), score) for keyword, score in keyword_yake if any(keyword.lower() in noun_chunk for noun_chunk in noun_chunks)]
    keyword_yake_noun_chunk = [(lemmatizer.lemmatize(keyword).lower() if len(keyword.split()) == 1 else " ".join([lemmatizer.lemmatize(word).lower() for word in keyword.split()]), score) for keyword, score in keyword_yake_noun_chunk]
    keyword_yake_noun_chunk = [(re.sub(r"[^a-zA-Z- ]", "", keyword).lower().strip(), score) for keyword, score in keyword_yake_noun_chunk]
    return keyword_yake_noun_chunk

def postprocess_keyword_list(keyword_yake_noun_chunk):
    keyword_yake_noun_chunk = [keyword for keyword, score in keyword_yake_noun_chunk[:10] if score <= 0.1]
    keyword_yake_noun_chunk = [keyword for keyword in keyword_yake_noun_chunk if keyword.isalnum() and len(keyword) >= 3]
    keyword_yake_noun_chunk = [keyword for keyword in keyword_yake_noun_chunk if not re.fullmatch(r'\b(?:[A-Z]{1,}\.){2,}\b|\b[A-Z]{1,3}\b', keyword)]
    keyword_yake_noun_chunk = [lemmatizer.lemmatize(keyword) for keyword in keyword_yake_noun_chunk if keyword not in stopwords]
    return keyword_yake_noun_chunk

def process_text_parallel(df, column, function):
    with mp.Pool(min(mp.cpu_count(), 12)) as pool:
        results = pool.map(function, df[column])
    df[column] = results
    return df

 

# Prepare Cleantech Data
df_cleantech = pd.read_json('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/df_keywords_list_agg_uspto_epo_rel_embeddings_noun_chunks.json')
df_cleantech.drop(columns=['keyword_yake_patentsberta_embedding', 'keyword_yake_climatebert_embedding', 'keyword_yake_bertforpatents_embedding'], inplace=True)
cleantech_list = df_cleantech['keyword_yake_lemma'].tolist()
patent_id_list = list(set(df_cleantech['patent_id'].explode().tolist()))
patent_id_list = [x for x in patent_id_list if str(x) != 'nan']
patent_id_list = patent_id_list[:1000]
publn_nr_list = list(set(df_cleantech['publn_nr'].explode().tolist()))
publn_nr_list = [x for x in publn_nr_list if str(x) != 'nan']
publn_nr_list = publn_nr_list[:1000]
oaid_list = list(set(df_cleantech['oaid'].explode().tolist()))
oaid_list = [x for x in oaid_list if str(x) != 'nan']
oaid_list = oaid_list[:1000]

Vectorizer = CountVectorizer(
    vocabulary = cleantech_list,
    ngram_range = (1, 4),
    stop_words='english',
    lowercase=True,
)

# Concatenate patent_id_list and publn_nr_list and oaid_list to g_cleantech
df_cleantech_patent_id_explode = df_cleantech.explode('patent_id')
df_cleantech_patent_id_explode = df_cleantech_patent_id_explode[['patent_id', 'keyword_yake_lemma']]
df_cleantech_patent_id_explode['patent_id'] = 'uspto-' + df_cleantech_patent_id_explode['patent_id']
df_cleantech_patent_id_explode = df_cleantech_patent_id_explode.rename(columns={'patent_id': 'id', 'keyword_yake_lemma': 'text'})
df_cleantech_patent_id_explode = df_cleantech_patent_id_explode.groupby('id')['text'].apply(' '.join).reset_index()
df_cleantech_patent_id_explode.reset_index(drop=True, inplace=True)

df_cleantech_publn_nr_explode = df_cleantech.explode('publn_nr')
df_cleantech_publn_nr_explode = df_cleantech_publn_nr_explode[['publn_nr', 'keyword_yake_lemma']]
df_cleantech_publn_nr_explode['publn_nr'] = 'epo-' + df_cleantech_publn_nr_explode['publn_nr']
df_cleantech_publn_nr_explode = df_cleantech_publn_nr_explode.rename(columns={'publn_nr': 'id', 'keyword_yake_lemma': 'text'})
df_cleantech_publn_nr_explode = df_cleantech_publn_nr_explode.groupby('id')['text'].apply(' '.join).reset_index()
df_cleantech_publn_nr_explode.reset_index(drop=True, inplace=True)

df_cleantech_oaid_explode = df_cleantech.explode('oaid')
df_cleantech_oaid_explode = df_cleantech_oaid_explode[['oaid', 'keyword_yake_lemma']]
df_cleantech_oaid_explode['oaid'] = 'rel-' + df_cleantech_oaid_explode['oaid']
df_cleantech_oaid_explode = df_cleantech_oaid_explode.rename(columns={'oaid': 'id', 'keyword_yake_lemma': 'text'})
df_cleantech_oaid_explode = df_cleantech_oaid_explode.groupby('id')['text'].apply(' '.join).reset_index()
df_cleantech_oaid_explode.reset_index(drop=True, inplace=True)

g_cleantech = pd.concat([df_cleantech_patent_id_explode, df_cleantech_publn_nr_explode, df_cleantech_oaid_explode], ignore_index=True)
g_cleantech_matrix = Vectorizer.transform(g_cleantech['text'])

del df_cleantech_patent_id_explode, df_cleantech_publn_nr_explode, df_cleantech_oaid_explode

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
    df_epo_non_cleantech_sample = process_text_parallel(df_epo_non_cleantech_sample, 'appln_text', clean_text)
    df_epo_non_cleantech_sample['epo_publn_nr'] = 'epo-' + df_epo_non_cleantech_sample['epo_publn_nr']
    df_epo_non_cleantech_sample = df_epo_non_cleantech_sample[['epo_publn_nr', 'appln_text']]
    df_epo_non_cleantech_sample.rename(columns={'epo_publn_nr': 'id', 'appln_text': 'text'}, inplace=True)
    df_epo_non_cleantech_sample = process_text_parallel(df_epo_non_cleantech_sample, 'text', extract_yake_keywords)
    df_epo_non_cleantech_sample = process_text_parallel(df_epo_non_cleantech_sample, 'text', postprocess_keyword_list)
    df_epo_non_cleantech_sample = df_epo_non_cleantech_sample[df_epo_non_cleantech_sample['text'].apply(lambda x: isinstance(x, str))]
    df_epo_non_cleantech_keyword_list = df_epo_non_cleantech_sample.explode('text').groupby('text')['id'].agg(['count', ('ids', lambda x: list(x))]).reset_index()

    # Randomly sample len(patent_id_list) patents from PatentsView Postgres database that are not in patent_id_list
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
        SELECT us_claims.patent_id, claim_text, claim_sequence
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
    df_uspto_non_cleantech_sample = process_text_parallel(df_uspto_non_cleantech_sample, 'claim_text', clean_text)
    df_uspto_non_cleantech_sample['patent_id'] = 'uspto-' + df_uspto_non_cleantech_sample['patent_id']
    df_uspto_non_cleantech_sample.rename(columns={'patent_id': 'id', 'claim_text': 'text'}, inplace=True)
    df_uspto_non_cleantech_sample = process_text_parallel(df_uspto_non_cleantech_sample, 'text', extract_yake_keywords)
    df_uspto_non_cleantech_sample = process_text_parallel(df_uspto_non_cleantech_sample, 'text', postprocess_keyword_list)
    df_uspto_non_cleantech_sample = df_uspto_non_cleantech_sample[df_uspto_non_cleantech_sample['text'].apply(lambda x: isinstance(x, str))]
    df_uspto_non_cleantech_keyword_list = df_uspto_non_cleantech_sample.explode('text').groupby('text')['id'].agg(['count', ('ids', lambda x: list(x))]).reset_index()

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
    df_rel_non_cleantech_sample['id'] = 'rel-' + df_rel_non_cleantech_sample['id']
    df_rel_non_cleantech_sample.dropna(subset=['abstract_inverted_index'], inplace=True)
    # Iterate over abstract_inverted_index column
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
    df_rel_non_cleantech_sample = process_text_parallel(df_rel_non_cleantech_sample, 'abstract', clean_text)
    df_rel_non_cleantech_sample.rename(columns={'id': 'oaid', 'abstract': 'text'}, inplace=True)
    df_rel_non_cleantech_sample = process_text_parallel(df_rel_non_cleantech_sample, 'text', extract_yake_keywords)
    df_rel_non_cleantech_sample = process_text_parallel(df_rel_non_cleantech_sample, 'text', postprocess_keyword_list)
    df_rel_non_cleantech_sample = df_rel_non_cleantech_sample[df_rel_non_cleantech_sample['text'].apply(lambda x: isinstance(x, str))]
    df_rel_non_cleantech_keyword_list = df_rel_non_cleantech_sample.explode('text').groupby('text')['oaid'].agg(['count', ('ids', lambda x: list(x))]).reset_index()

    # Concatenate non_cleantech_keyword_list dataframes
    df_non_cleantech_keyword_list = pd.concat([df_epo_non_cleantech_keyword_list, df_uspto_non_cleantech_keyword_list, df_rel_non_cleantech_keyword_list], ignore_index=True)
    # Print a keyword longer than 2 words
    print(f"Non Cleantech Keyword List head: {df_non_cleantech_keyword_list[df_non_cleantech_keyword_list['text'].str.split().str.len() > 2].sample(5)}")
    df_non_cleantech_keyword_list = df_non_cleantech_keyword_list.groupby('text')['count'].sum().reset_index()
    df_non_cleantech_keyword_list = df_non_cleantech_keyword_list[(df_non_cleantech_keyword_list['count'] >= 5) & (df_non_cleantech_keyword_list['count'] <= 1000)]
    print(f"Non Cleantech Keyword List head: {df_non_cleantech_keyword_list[df_non_cleantech_keyword_list['text'].str.split().str.len() > 2].sample(5)}")


    # Concatenate samples and perform Logistic Regression
    g_non_cleantech = pd.concat([df_epo_non_cleantech_sample, df_uspto_non_cleantech_sample, df_rel_non_cleantech_sample], ignore_index=True)
    g_non_cleantech['text'] = g_non_cleantech['text'].apply(lambda x: ' '.join([keyword for keyword in x if keyword in df_non_cleantech_keyword_list['text'].tolist()]))
    print(f"Non Cleantech Sample head: {g_non_cleantech[g_non_cleantech['text'].str.split().str.len() > 2].sample(5)}")
    g_non_cleantech_matrix = Vectorizer.transform(g_non_cleantech['text'])

    # Concatenate data for train_test_split
    X = vstack([g_cleantech_matrix, g_non_cleantech_matrix])
    y = np.concatenate([np.ones(g_cleantech_matrix.shape[0]), np.zeros(g_non_cleantech_matrix.shape[0])])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and evaluate the model
    df_cleantech = train_evaluate_model(i, model, X_train, X_test, y_train, y_test, df_cleantech, list_classification_reports)

    print(df_cleantech.head())
    print(list_classification_reports)
    print(f"Finished iteration {i+1} of {MAX_ITER}, classification report: {list_classification_reports[i]}")

conn_openalex.close()
conn_patstat.close()