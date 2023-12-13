import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import re
import nltk
from nltk.stem import WordNetLemmatizer
# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

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
    text = re.sub(r"[^a-zA-Z- ]", " ", text).lower().strip()

    # Lemmatize each word
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return lemmatized_text


### Prepare Cleantech Data
# Co-Occurrence Directory
# co_occurrence_dir = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Co-Occurrence Analysis/'
# co_occurrence_files = glob.glob(co_occurrence_dir + '*.csv')
co_occurrence_files = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Co-Occurrence Analysis/co_occurrence_matrix_yake_keywords_cleantech_uspto_epo_rel_ids_semantic_similarity_02.csv'

# Similarity Directory
# similarity_dir = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Similarity Search/'
# similarity_files = glob.glob(similarity_dir + '*.json')
similarity_files = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Similarity Search/df_keyword_titles_cosine_similarity_radius_025_neighbors_100_noun_chunks.json'

# Co-Occurrence Threshold
# co_occurrence_threshold = [0.01, 0.025, 0.05, 0.1, 0.15]
co_occurrence_threshold = [0.01]

# Load the data
df_cleantech_cooccurrence = pd.read_csv(co_occurrence_files, index_col=0)
df_cleantech_cooccurrence.dropna(how='all', inplace=True)

df_cleantech_similarity = pd.read_json(similarity_files)

# Co-Occurrence Threshold
co_occurrence_threshold = 0.01  # Assuming you are using a single threshold value

# Create a mask for the co-occurrence threshold
mask = df_cleantech_cooccurrence.applymap(lambda x: x >= co_occurrence_threshold)

# Apply mask to DataFrame
filtered_co_occurrence_df = df_cleantech_cooccurrence[mask]

# Extract keywords
co_occurrence_list = filtered_co_occurrence_df.columns[filtered_co_occurrence_df.any()].tolist()

# Processing similarity data
similarity_series = pd.concat([df_cleantech_similarity['keyword_yake_lemma'], df_cleantech_similarity['keywords_keyword_yake_bertforpatents_embedding'].explode()], ignore_index=True)
similarity_list = similarity_series.drop_duplicates().tolist()

# Combine and deduplicate lists
cleantech_list = list(set(co_occurrence_list + similarity_list))
cleantech_list = [str(keyword) for keyword in cleantech_list]

# # Create DataFrame
# df_cleantech = pd.DataFrame(cleantech_list, columns=['keyword_yake_lemma'])
# df_cleantech['cleantech'] = 1

del df_cleantech_cooccurrence
del df_cleantech_similarity
del co_occurrence_list
del similarity_list

g_uspto_cleantech = pd.read_json('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/g_patent_claims_fulltext_cleantech.json')
g_uspto_cleantech['patent_id'] = 'us-' + g_uspto_cleantech['patent_id'].astype(str)
# Delete all None values in the claim_fulltext column
g_uspto_cleantech = g_uspto_cleantech[g_uspto_cleantech['claim_fulltext'].notna()]
g_uspto_cleantech['claim_fulltext'] = g_uspto_cleantech['claim_fulltext'].astype(str)
g_uspto_cleantech['claim_fulltext'] = g_uspto_cleantech['claim_fulltext'].apply(clean_and_lemmatize)
# Create Document List for CountVectorizer
document_list = g_uspto_cleantech['claim_fulltext'].tolist()
# Cast to string
document_list = [str(x) for x in document_list]
# Create CountVectorizer
vectorizer_uspto_cleantech = CountVectorizer(vocabulary=cleantech_list)
# Create Document Term Matrix
document_term_matrix_uspto_cleantech = vectorizer_uspto_cleantech.fit_transform(document_list)
# Create DataFrame
df_uspto_cleantech = pd.DataFrame(document_term_matrix_uspto_cleantech.toarray().transpose(),
                                  index=vectorizer_uspto_cleantech.get_feature_names_out(),
                                  columns=g_uspto_cleantech['patent_id'])
# Save DataFrame
# df_uspto_cleantech.to_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_uspto_cleantech.csv')
df_uspto_cleantech.to_parquet('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_uspto_cleantech.parquet')

# Delete all variables
del df_uspto_cleantech
del document_list
del vectorizer_uspto_cleantech
del document_term_matrix_uspto_cleantech
del g_uspto_cleantech

g_uspto_non_cleantech = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/g_uspto_non_cleantech_claims_fulltext.json')
g_uspto_non_cleantech['patent_id'] = 'us-' + g_uspto_non_cleantech['patent_id'].astype(str)
# Delete all None values in the claim_fulltext column
g_uspto_non_cleantech = g_uspto_non_cleantech[g_uspto_non_cleantech['claim_fulltext'].notna()]
g_uspto_non_cleantech['claim_fulltext'] = g_uspto_non_cleantech['claim_fulltext'].astype(str)
g_uspto_non_cleantech['claim_fulltext'] = g_uspto_non_cleantech['claim_fulltext'].apply(clean_and_lemmatize)
# Create Document List for CountVectorizer
document_list = g_uspto_non_cleantech['claim_fulltext'].tolist()
# Cast to string
document_list = [str(x) for x in document_list]
# Create CountVectorizer
vectorizer_uspto_non_cleantech = CountVectorizer(vocabulary=cleantech_list)
# Create Document Term Matrix
document_term_matrix_uspto_non_cleantech = vectorizer_uspto_non_cleantech.fit_transform(document_list)
# Create DataFrame
df_uspto_non_cleantech = pd.DataFrame(document_term_matrix_uspto_non_cleantech.toarray().transpose(),
                                  index=vectorizer_uspto_non_cleantech.get_feature_names_out(),
                                  columns=g_uspto_non_cleantech['patent_id'])
# Save DataFrame
# df_uspto_non_cleantech.to_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_uspto_non_cleantech.csv')
df_uspto_non_cleantech.to_parquet('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_uspto_non_cleantech.parquet')

# Delete all variables
del df_uspto_non_cleantech
del document_list
del vectorizer_uspto_non_cleantech
del document_term_matrix_uspto_non_cleantech
del g_uspto_non_cleantech

g_epo_cleantech = pd.read_json('/mnt/hdd01/PATSTAT Working Directory/PATSTAT/cleantech_epo_text_data_pivot_cleaned.json')
g_epo_cleantech['publn_nr'] = 'ep-' + g_epo_cleantech['publn_nr'].astype(str)
# Delete all None values in the cleaned_claims column
g_epo_cleantech = g_epo_cleantech[g_epo_cleantech['cleaned_claims'].notna()]
g_epo_cleantech['cleaned_claims'] = g_epo_cleantech['cleaned_claims'].astype(str)
g_epo_cleantech['cleaned_claims'] = g_epo_cleantech['cleaned_claims'].apply(clean_and_lemmatize)
# Create Document List for CountVectorizer
document_list = g_epo_cleantech['cleaned_claims'].tolist()
# Cast to string
document_list = [str(x) for x in document_list]
# Create CountVectorizer
vectorizer_epo_cleantech = CountVectorizer(vocabulary=cleantech_list)
# Create Document Term Matrix
document_term_matrix_epo_cleantech = vectorizer_epo_cleantech.fit_transform(document_list)
# Create DataFrame
df_epo_cleantech = pd.DataFrame(document_term_matrix_epo_cleantech.toarray().transpose(),
                                  index=vectorizer_epo_cleantech.get_feature_names_out(),
                                  columns=g_epo_cleantech['publn_nr'])
# Save DataFrame
# df_epo_cleantech.to_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_epo_cleantech.csv')
df_epo_cleantech.to_parquet('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_epo_cleantech.parquet')

# Delete all variables
del df_epo_cleantech
del document_list
del vectorizer_epo_cleantech
del document_term_matrix_epo_cleantech
del g_epo_cleantech

g_epo_non_cleantech = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_epo_non_cleantech_text_data_pivot_claims_cleaned.json')
g_epo_non_cleantech['publn_nr'] = 'ep-' + g_epo_non_cleantech['publn_nr'].astype(str)
# Delete all None values in the cleaned_claims column
g_epo_non_cleantech = g_epo_non_cleantech[g_epo_non_cleantech['cleaned_claims'].notna()]
g_epo_non_cleantech['cleaned_claims'] = g_epo_non_cleantech['cleaned_claims'].astype(str)
g_epo_non_cleantech['cleaned_claims'] = g_epo_non_cleantech['cleaned_claims'].apply(clean_and_lemmatize)
# Create Document List for CountVectorizer
document_list = g_epo_non_cleantech['cleaned_claims'].tolist()
# Cast to string
document_list = [str(x) for x in document_list]
# Create CountVectorizer
vectorizer_epo_non_cleantech = CountVectorizer(vocabulary=cleantech_list)
# Create Document Term Matrix
document_term_matrix_epo_non_cleantech = vectorizer_epo_non_cleantech.fit_transform(document_list)
# Create DataFrame
df_epo_non_cleantech = pd.DataFrame(document_term_matrix_epo_non_cleantech.toarray().transpose(),
                                  index=vectorizer_epo_non_cleantech.get_feature_names_out(),
                                  columns=g_epo_non_cleantech['publn_nr'])
# Save DataFrame
# df_epo_non_cleantech.to_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_epo_non_cleantech.csv')
df_epo_non_cleantech.to_parquet('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_epo_non_cleantech.parquet')

# Delete all variables
del df_epo_non_cleantech
del document_list
del vectorizer_epo_non_cleantech
del document_term_matrix_epo_non_cleantech
del g_epo_non_cleantech

g_rel_cleantech = pd.read_json('/mnt/hdd01/patentsview/Reliance on Science - Cleantech Patents/df_oaid_cleantech_yake_noun_chunks.json')
g_rel_cleantech['oaid'] = 'rel-' + g_rel_cleantech['oaid'].astype(str)
# Delete all None values in the abstract column
g_rel_cleantech = g_rel_cleantech[g_rel_cleantech['abstract'].notna()]
g_rel_cleantech['abstract'] = g_rel_cleantech['abstract'].astype(str)
g_rel_cleantech['abstract'] = g_rel_cleantech['abstract'].apply(clean_and_lemmatize)
# Create Document List for CountVectorizer
document_list = g_rel_cleantech['abstract'].tolist()
# Cast to string
document_list = [str(x) for x in document_list]
# Create CountVectorizer
vectorizer_rel_cleantech = CountVectorizer(vocabulary=cleantech_list)
# Create Document Term Matrix
document_term_matrix_rel_cleantech = vectorizer_rel_cleantech.fit_transform(document_list)
# Create DataFrame
df_rel_cleantech = pd.DataFrame(document_term_matrix_rel_cleantech.toarray().transpose(),
                                  index=vectorizer_rel_cleantech.get_feature_names_out(),
                                  columns=g_rel_cleantech['oaid'])
# Save DataFrame
# df_rel_cleantech.to_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_rel_cleantech.csv')
df_rel_cleantech.to_parquet('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_rel_cleantech.parquet')

# Delete all variables
del df_rel_cleantech
del document_list
del vectorizer_rel_cleantech
del document_term_matrix_rel_cleantech
del g_rel_cleantech

g_rel_non_cleantech = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/df_oaids_non_cleantech.json')
g_rel_non_cleantech['oaid'] = 'rel-' + g_rel_non_cleantech['oaid'].astype(str)
# Delete all None values in the abstract column
g_rel_non_cleantech = g_rel_non_cleantech[g_rel_non_cleantech['abstract'].notna()]
g_rel_non_cleantech['abstract'] = g_rel_non_cleantech['abstract'].astype(str)
g_rel_non_cleantech['abstract'] = g_rel_non_cleantech['abstract'].apply(clean_and_lemmatize)
# Create Document List for CountVectorizer
document_list = g_rel_non_cleantech['abstract'].tolist()
# Cast to string
document_list = [str(x) for x in document_list]
# Create CountVectorizer
vectorizer_rel_non_cleantech = CountVectorizer(vocabulary=cleantech_list)
# Create Document Term Matrix
document_term_matrix_rel_non_cleantech = vectorizer_rel_non_cleantech.fit_transform(document_list)
# Create DataFrame
df_rel_non_cleantech = pd.DataFrame(document_term_matrix_rel_non_cleantech.toarray().transpose(),
                                  index=vectorizer_rel_non_cleantech.get_feature_names_out(),
                                  columns=g_rel_non_cleantech['oaid'])
# Save DataFrame
# df_rel_non_cleantech.to_csv('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_rel_non_cleantech.csv')
df_rel_non_cleantech.to_parquet('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/TFIDF Matrices/df_rel_non_cleantech.parquet')

# Delete all variables
del df_rel_non_cleantech
del document_list
del vectorizer_rel_non_cleantech
del document_term_matrix_rel_non_cleantech
del g_rel_non_cleantech