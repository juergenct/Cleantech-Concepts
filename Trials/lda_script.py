import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Import test data
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_sample_keyphrase.csv')

# Concatenate abstracts for the same value in the 'cpc_subgroup' column - SHOULD I REALLY DO THIS??
df = df.groupby('cpc_subgroup')['patent_abstract'].apply(' '.join).reset_index()
# # Remove duplicate rows on the 'cpc_subgroup' column
# df = df.drop_duplicates(subset=['cpc_subgroup'])

# Set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Tokenize and preprocess the text
texts = []
for index, row in df.iterrows():
    tokens = word_tokenize(row['patent_abstract'])
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    texts.append(tokens)

# Create a dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train the LDA model
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)

# Extract keywords from the LDA model
keywords_list = []
for index, row in df.iterrows():
    tokens = word_tokenize(row['patent_abstract'])
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    bow = dictionary.doc2bow(tokens)
    topic_distribution = lda_model.get_document_topics(bow)
    keywords = [dictionary[word_id] for word_id, _ in topic_distribution]
    keywords_list.append(', '.join(keywords))

# Add the keywords to the dataframe
df['keywords'] = keywords_list

# Save dataframe to JSON
df.to_json('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/LDA/df_sample_keyphrase.json')
