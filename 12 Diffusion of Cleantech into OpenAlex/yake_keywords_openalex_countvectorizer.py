from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import gcld3

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize language detector
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

num_works = 400000 #400000 # Goal is 100000

# List of OpenAlex 0 Level Concepts
concept_level_0 = {
    'Biology': 'https://openalex.org/C86803240',
    'Medicine': 'https://openalex.org/C71924100',
    'Chemistry': 'https://openalex.org/C185592680',
    'Computer science': 'https://openalex.org/C41008148',
    'Physics': 'https://openalex.org/C121332964',
    'Materials science': 'https://openalex.org/C192562407',
    'Engineering': 'https://openalex.org/C127413603',
    'Psychology': 'https://openalex.org/C15744967',
    'Economics': 'https://openalex.org/C162324750',
    'Philosophy': 'https://openalex.org/C138885662',
    'Political science': 'https://openalex.org/C17744445',
    'Sociology': 'https://openalex.org/C144024400',
    'Geology': 'https://openalex.org/C127313418',
    'Environmental science': 'https://openalex.org/C39432304',
    'Business': 'https://openalex.org/C144133560',
    'History': 'https://openalex.org/C95457728',
    'Art': 'https://openalex.org/C142362112',
    'Mathematics': 'https://openalex.org/C33923547',
    'Geography': 'https://openalex.org/C205649164'
}

# Database credentials and connection parameters
database_url = URL.create(
    drivername="postgresql+psycopg2",
    username='tie',
    password='TIE%2023!tuhh',
    host='100.113.100.152',
    port=45432,
    database='openalex_db'
)

# Create an engine
engine = create_engine(database_url)

# Load cleantech keywords
df_cleantech_keywords = pd.read_json("/home/thiesen/Documents/Cleantech_Concepts/cleantech_keywords_similarity_015_co_occurrence_025_claim_fulltext.json")
cleantech_keywords = df_cleantech_keywords['keyword_yake_lemma'].tolist()
vocabulary = set(cleantech_keywords)  # Make sure the vocabulary is unique

# Load rel_on_science
df_rel_on_science = pd.read_json('/mnt/hdd01/patentsview/Reliance on Science - Cleantech Patents/df_oaid_cleantech_lang_detect_yake_title_abstract_noun_chunks.json')
oaid_cleantech_id = df_rel_on_science['id'].tolist()

# Use CountVectorizer with the predefined vocabulary
vectorizer = CountVectorizer(vocabulary=vocabulary)

# Loop through each concept
for concept_name, concept_id in concept_level_0.items():
    print(f"Processing {concept_name}...")

    # Define the query
    query_concept = f"""
        SELECT w.id, w.title, w.abstract_inverted_index, w.publication_year
        FROM openalex.works w
        JOIN (
            SELECT wc.work_id
            FROM openalex.works_concepts wc
            WHERE wc.concept_id = '{concept_id}'
            AND wc.score > 0.70
        ) AS filtered_works ON w.id = filtered_works.work_id
        WHERE w.publication_year BETWEEN 1950 AND 2023
        ORDER BY RANDOM()
        LIMIT {num_works};
    """

    # Execute query
    result_work_text = pd.read_sql_query(query_concept, engine)

    # Delete all entries where id is NaN
    result_work_text = result_work_text.dropna(subset=['id'])

    # Drop all result_work_text where id is in oaid_cleantech_id
    result_work_text = result_work_text[~result_work_text['id'].isin(oaid_cleantech_id)]

    # Iterate over abstract_inverted_index column
    for index, row in result_work_text.iterrows():
        word_index = []
        try:
            for key, value in row['abstract_inverted_index'].items():
                if key == 'InvertedIndex':
                    for innerkey, innervalue in value.items():
                        # Lemmatize the word before appending
                        lemma = lemmatizer.lemmatize(innerkey.lower())
                        for innerindex in innervalue:
                            word_index.append([lemma, innerindex])
            # Sort list by index
            word_index.sort(key=lambda x: x[1])
            # Join first element of each list in word_index
            abstract = ' '.join([i[0] for i in word_index])
            # Add column abstract to result dataframe
            result_work_text.at[index, 'abstract'] = abstract
            # Check whether the abstract is in English, if not delete the row
            if detector.FindLanguage(abstract).language != 'en':
                result_work_text.drop(index, inplace=True)
        except AttributeError:
            continue
    
    # Concatenate 'title' and 'abstract'
    result_work_text['document'] = result_work_text['title'] + ' ' + result_work_text['abstract']
    result_work_text = result_work_text.dropna(subset=['document'])

    # Limit the dataframe to the first 100000 rows
    result_work_text = result_work_text.dropna(subset=['id'])
    result_work_text = result_work_text.head(100000)

    # Log if the dataframe has less than 100000 rows
    if len(result_work_text) < 100000:
        print(f"Dataframe has less than 100000 rows for concept {concept_id}: {len(result_work_text)}")

    # Preprocess 'keyword_yake_lemma' to create vocabulary
    cleantech_keywords = df_cleantech_keywords['keyword_yake_lemma'].tolist()
    vocabulary = set(cleantech_keywords)  # Make sure the vocabulary is unique

    # Document Term Matrix
    dtm = vectorizer.fit_transform(result_work_text['document'])
    dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Concatenate the DTM DataFrame with the result_work_text DataFrame
    result_work_text = pd.concat([result_work_text, dtm_df], axis=1)

    # Save to CSV, naming the file after the concept
    file_path = f'/home/thiesen/Documents/Cleantech_Concepts/{concept_name.replace(" ", "_")}_result_work_text_openalex.csv'
    result_work_text.to_csv(file_path)
    print(f"Saved {concept_name} results to {file_path}")

    del result_work_text, dtm, dtm_df
    
# Close the engine
engine.dispose()