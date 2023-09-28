import os
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')
docs = df['patent_abstract'].tolist()
labels = df['concept'].tolist()

# Initialize BERTopic Models
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression(random_state=42, max_iter=1000)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

topic_model_supervised = BERTopic(
                                umap_model=empty_dimensionality_model,
                                hdbscan_model=clf,
                                ctfidf_model=ctfidf_model,
)

topics, probs = topic_model_supervised.fit_transform(docs, labels)

# Save Models
topic_model_supervised.save('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/Bertopic/Supervised', serialization='pytorch')
