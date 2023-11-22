import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sentence_transformers import SentenceTransformer
import torch

df_keywords_list_agg_pruned = pd.read_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/uspto_epo_rel_keywords_list_non_cleantech_noun_chunks_processed.json')

model_climatebert = SentenceTransformer('climatebert/distilroberta-base-climate-f')
model_bertforpatents = SentenceTransformer('anferico/bert-for-patents')
model_patentsberta = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Generate copy of df_claims_keywords_list
df_keywords_list_agg_embeddings = df_keywords_list_agg_pruned.copy()

# Perform sentence embedding on the 'keyword_yake' (PatentsView) or 'keywords_yake_claims' (EPO) column
df_keywords_list_agg_embeddings['keyword_yake_patentsberta_embedding'] = df_keywords_list_agg_embeddings['keyword_yake_lemma'].progress_apply(
    lambda x: model_patentsberta.encode(x)
)

df_keywords_list_agg_embeddings['keyword_yake_climatebert_embedding'] = df_keywords_list_agg_embeddings['keyword_yake_lemma'].progress_apply(
    lambda x: model_climatebert.encode(x)
)

df_keywords_list_agg_embeddings['keyword_yake_bertforpatents_embedding'] = df_keywords_list_agg_embeddings['keyword_yake_lemma'].progress_apply(
    lambda x: model_bertforpatents.encode(x)
)

# Save df_claims_keywords_list_embeddings to json
df_keywords_list_agg_embeddings.to_json('/mnt/hdd01/patentsview/Non Cleantech Patents - Classifier Set/uspto_epo_rel_keywords_list_non_cleantech_noun_chunks_processed_embeddings.json')