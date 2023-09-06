import os
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance, ZeroShotClassification
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')

# Initialize BERTopic Models
embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
umap_model = UMAP(n_neighbors=15)
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')
# cluster_model = KMeans(n_clusters=41, init='k-means++', random_state=42)
vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
ctfidf_model = ClassTfidfTransformer()

representation_model_KeyBERT = KeyBERTInspired()
representation_model_MMR = MaximalMarginalRelevance(diversity=0.7) # Try out with MMR and rather high diversity 0.7, might try with lower diversity
representation_model_chain = [representation_model_MMR, representation_model_KeyBERT]
# representation_model_ZeroShot = ZeroShotClassification(candidate_labels, model='LLM') # Maybe try out with ZeroShotClassification and set labels

# pos_patterns = [{"NP": ["NOUN", "PROPN"] - Maybe introduce Noun Phrase Extraction here
# representation_model_POS = PartOfSpeech('en_core_web_sm', pos_patterns=pos_patterns)

# Train Models
topic_model_KeyBERT = BERTopic(embedding_model=embedding_model,
                                umap_model=umap_model,
                                hdbscan_model=hdbscan_model,
                                vectorizer_model=vectorizer_model,
                                ctfidf_model=ctfidf_model,
                                representation_model=representation_model_KeyBERT)
topics_KeyBERT, probs_KeyBERT = topic_model_KeyBERT.fit_transform(df['patent_abstract'])

topic_model_MMR = BERTopic(embedding_model=embedding_model,
                            umap_model=umap_model,
                            hdbscan_model=hdbscan_model,
                            vectorizer_model=vectorizer_model,
                            ctfidf_model=ctfidf_model,
                            representation_model=representation_model_MMR)
topics_MMR, probs_MMR = topic_model_MMR.fit_transform(df['patent_abstract'])

topic_model_chain = BERTopic(embedding_model=embedding_model,
                            umap_model=umap_model,
                            hdbscan_model=hdbscan_model,
                            vectorizer_model=vectorizer_model,
                            ctfidf_model=ctfidf_model,
                            representation_model=representation_model_chain)
topics_chain, probs_chain = topic_model_chain.fit_transform(df['patent_abstract'])

# Save Models
topic_model_KeyBERT.save('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/Bertopic/KeyBERT', serialization='pytorch')
topic_model_MMR.save('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/Bertopic/MMR', serialization='pytorch')
topic_model_chain.save('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/Bertopic/Chain_KeyBERT_MMR', serialization='pytorch')
