import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
import torch

### Prepare Non-Cleantech Data
df_non_cleantech_raw = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/uspto_epo_rel_keywords_list_non_cleantech_noun_chunks_processed_embeddings.json')
df_non_cleantech_raw['cleantech'] = 0
# Drop all columns except keyword_yake_lemma and cleantech
df_non_cleantech_raw.drop(columns=df_non_cleantech_raw.columns.difference(['keyword_yake_lemma', 'cleantech']), inplace=True)

### Prepare Cleantech Data
# Co-Occurrence Directory
co_occurrence_dir = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Co-Occurrence Analysis/'
co_occurrence_files = glob.glob(co_occurrence_dir + '*.csv')

# Similarity Directory
similarity_dir = '/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Similarity Search/'
similarity_files = glob.glob(similarity_dir + '*.json')

# Co-Occurrence Threshold
# co_occurrence_threshold = [0.01, 0.025, 0.05, 0.1, 0.15]
co_occurrence_threshold = [0.1]

# Result Dataframe
df_results_svc = pd.DataFrame()
# Columns co_occurrence file, similarity file, len cleantech, co_occurrence_threshold, classification_report
df_results_svc['co_occurrence_file'] = ''
df_results_svc['similarity_file'] = ''
df_results_svc['co_occurrence_threshold'] = ''
df_results_svc['classification_report'] = ''
df_results_svc['len_cleantech'] = ''
df_results_svc['len_similarity_list'] = ''
df_results_svc['len_co_occurrence_list'] = ''

# Misc Variables
model_bertforpatents = SentenceTransformer('anferico/bert-for-patents')
# Check if GPU is available
if torch.cuda.is_available():
    # Move model to GPU
    model_bertforpatents.to('cuda')
df_non_cleantech_raw['keyword_yake_lemma_bertforpatents_embedding'] = model_bertforpatents.encode(df_non_cleantech_raw['keyword_yake_lemma'].tolist()).tolist()


for co_file in co_occurrence_files:
    # Import Co-Occurrence Cleantech data
    df_cleantech_cooccurrence = pd.read_csv(co_file, index_col=0)
    # Delete all rows where all values are NaN
    df_cleantech_cooccurrence.dropna(how='all', inplace=True)
    for sim_file in similarity_files:
        # Import Similarity Search Cleantech data
        df_cleantech_similarity = pd.read_json(sim_file)
        for co_threshold in co_occurrence_threshold:
            try:
                print(f"Co-Occurrence File: {co_file}")
                print(f"Similarity File: {sim_file}")
                print(f"Co-Occurrence Threshold: {co_threshold}")

                # Get boolean mask where any value in column is greater than or equal to co_threshold
                mask = (df_cleantech_cooccurrence >= co_threshold).any()
                # Apply mask to columns and convert to list
                co_occurrence_list = df_cleantech_cooccurrence.columns[mask].tolist()
                # Drop duplicates
                co_occurrence_list = list(dict.fromkeys(co_occurrence_list))
                # Concatenate the two columns into a single Series
                df_cleantech_similarity.columns
                similarity_series = pd.concat([df_cleantech_similarity['keyword_yake_lemma'], df_cleantech_similarity['keywords_keyword_yake_bertforpatents_embedding'].explode()], ignore_index=True)
                # Drop duplicates and convert to list
                similarity_list = similarity_series.drop_duplicates().tolist()
                cleantech_list = list(dict.fromkeys(co_occurrence_list + similarity_list))
                # Drop duplicates
                cleantech_list = list(dict.fromkeys(cleantech_list))

                df_cleantech = pd.DataFrame(cleantech_list, columns=['keyword_yake_lemma'])
                df_cleantech['cleantech'] = 1
                df_cleantech = df_cleantech[df_cleantech['keyword_yake_lemma'].apply(lambda x: isinstance(x, str))]
                df_cleantech['keyword_yake_lemma_bertforpatents_embedding'] = model_bertforpatents.encode(df_cleantech['keyword_yake_lemma'].tolist()).tolist()

                print(f"Number of Cleantech Keywords: {len(df_cleantech)}")

                df_cleantech.to_json('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/cleantech_keywords_similarity_015_co_occurrence_01.json', orient='records')

                # Randomly sample non-cleantech data, len = len(cleantech)
                df_non_cleantech = df_non_cleantech_raw.sample(n=len(df_cleantech), random_state=42)

                # Concatenate dataframes
                df = pd.concat([df_cleantech, df_non_cleantech], ignore_index=True)

                ### Perform Classification
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(df['keyword_yake_lemma_bertforpatents_embedding'], df['cleantech'], test_size=0.2, shuffle=True, random_state=42)

                # Convert to numpy array
                X_train = np.array(X_train.tolist())
                X_test = np.array(X_test.tolist())

                # Train SVM
                clf = SVC(random_state=42, cache_size=10000)
                clf.fit(X_train, y_train)

                # Create new row
                new_row = {'co_occurrence_file': co_file, 'similarity_file': sim_file, 'co_occurrence_threshold': co_threshold, 'classification_report': classification_report(y_test, clf.predict(X_test), output_dict=True), 'len_cleantech': len(df_cleantech), 'len_similarity_list': len(similarity_list), 'len_co_occurrence_list': len(co_occurrence_list)}
                # Append row to the dataframe
                df_results_svc = pd.concat([df_results_svc, pd.DataFrame([new_row])], ignore_index=True)

                # Print classification report
                print(classification_report(y_test, clf.predict(X_test)))
            except:
                print(f"Error: {co_file}, {sim_file}, {co_threshold}")

# Save results
# df_results_svc.to_json('/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/svc_classification_cleantech_dictionary_neighbors_and_radius.json', orient='records')
