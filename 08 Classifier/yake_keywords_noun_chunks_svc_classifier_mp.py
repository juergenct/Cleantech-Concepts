import glob
import os
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map  # For progress bar with multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"
multiprocessing.set_start_method('spawn', force=True)

column = 'title' # 'title', 'abstract' or 'claim_fulltext'

def process_files(args):
    co_file, sim_file, co_threshold, df_non_cleantech_raw, model_bertforpatents = args

    # Import Cleantech data
    df_cleantech_similarity = pd.read_json(sim_file)
    df_cleantech_cooccurrence = pd.read_csv(co_file, index_col=0)
    df_cleantech_cooccurrence.dropna(how='all', inplace=True)

    mask = (df_cleantech_cooccurrence >= co_threshold).any()
    co_occurrence_list = df_cleantech_cooccurrence.columns[mask].tolist()

    similarity_series = df_cleantech_similarity['keyword_yake_lemma'].append(df_cleantech_similarity['keywords_keyword_yake_bertforpatents_embedding'].explode())
    similarity_list = similarity_series.drop_duplicates().tolist()

    cleantech_list = co_occurrence_list + similarity_list
    df_cleantech = pd.DataFrame(cleantech_list, columns=['keyword_yake_lemma'])
    df_cleantech['cleantech'] = 1
    df_cleantech['keyword_yake_lemma'] = model_bertforpatents.encode(df_cleantech['keyword_yake_lemma'].tolist()).tolist()

    df_non_cleantech = df_non_cleantech_raw.sample(n=len(df_cleantech), random_state=42)
    df = pd.concat([df_cleantech, df_non_cleantech], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df['keyword_yake_lemma'], df['cleantech'], test_size=0.2, random_state=42)
    X_train = np.array(X_train.tolist())
    X_test = np.array(X_test.tolist())

    clf = SVC(random_state=42)
    clf.fit(X_train, y_train)

    return {
        'co_occurrence_file': co_file,
        'similarity_file': sim_file,
        'co_occurrence_threshold': co_threshold,
        'len_cleantech': len(df_cleantech),
        'classification_report': classification_report(y_test, clf.predict(X_test), output_dict=True)
    }

# Prepare Non-Cleantech Data
df_non_cleantech_raw = pd.read_json(f'/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/uspto_epo_rel_keywords_list_non_cleantech_{column}_noun_chunks_processed_embeddings.json', orient='records')
df_non_cleantech_raw['cleantech'] = 0
df_non_cleantech_raw.drop(columns=df_non_cleantech_raw.columns.difference([f'keywords_yake_{column}_lemma', 'cleantech']), inplace=True)
df_non_cleantech_raw.rename(columns={f'keywords_yake_{column}_lemma': 'keyword_yake_lemma'}, inplace=True)

# Prepare model
model_bertforpatents = SentenceTransformer('anferico/bert-for-patents')
if torch.cuda.is_available():
    model_bertforpatents.to('cuda')
df_non_cleantech_raw['keyword_yake_lemma_bertforpatents_embedding'] = model_bertforpatents.encode(df_non_cleantech_raw[f'keyword_yake_lemma'].tolist()).tolist()

# File paths
co_occurrence_files = glob.glob(f'/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Co-Occurrence Analysis/co_occurrence_matrix_yake_keywords_cleantech_uspto_epo_rel_ids_semantic_similarity_{column}.csv')
similarity_files = glob.glob(f"/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Similarity Search/*.json")
# f"/mnt/hdd01/patentsview/Similarity Search - CPC Classification and Claims/Similarity Search/df_keyword_{column}s_cosine_similarity_radius_{str(radius).replace('.','')}_neighbors_{str(k_neighbors)}_noun_chunks.json"
co_occurrence_threshold = [0.01, 0.025, 0.05, 0.1, 0.15]

print(f'Co-Occurence Files: {co_occurrence_files}')
print(f'Similarity Files: {similarity_files}')

# Prepare arguments for multiprocessing
task_args = [(co_file, sim_file, co_threshold, df_non_cleantech_raw, model_bertforpatents)
             for co_file in co_occurrence_files
             for sim_file in similarity_files
             for co_threshold in co_occurrence_threshold]

# Multiprocessing with progress bar
results = process_map(process_files, task_args, max_workers=8, chunksize=1)

# Convert results to DataFrame and save
df_results_svc = pd.DataFrame(results)
df_results_svc.to_json(f'/home/thiesen/Documents/Cleantech_Concepts/df_svc_yake_keywords_{column}.json', orient='records')
