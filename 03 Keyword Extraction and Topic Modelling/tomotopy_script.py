import os
import re
import json
import pandas as pd
import tomotopy as tp
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet

# Load data
df = pd.read_json('/home/thiesen/Documents/Cleantech_Concepts/g_patent_claims_cleantech_test.json')

# Split data into train and test set
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

# Function to obtain part-of-speech tags for each word in corpus (only valid options for lemmatizer)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()

# Initialize model
PLDA_model_IDF = tp.PLDAModel(tw=tp.TermWeight.IDF, topics_per_label=10, seed=42, latent_topics=1)
PLDA_model_ONE = tp.PLDAModel(tw=tp.TermWeight.ONE, topics_per_label=10, seed=42, latent_topics=1)
PLDA_model_PMI = tp.PLDAModel(tw=tp.TermWeight.PMI, topics_per_label=10, seed=42, latent_topics=1)

# Add documents to model
for index, row in tqdm(df_train.iterrows()):
    # Lowercase all words
    clean_document = row['claim_fulltext'].lower()

    ### Regex ###
    # Replace '-' and '/' with spaces
    clean_document = re.sub(r'[-/]', ' ', clean_document)
    # Remove puncutation
    clean_document = re.sub(r'[^\w\s]', '', clean_document)
    # Remove numbers
    clean_document = re.sub(r'\d+', '', clean_document)
    # # Remove words with less than 2 characters
    # clean_document = re.sub(r'\b\w{1,3}\b', '', clean_document)
    # Remove extra spaces
    clean_document = re.sub(r'\s+', ' ', clean_document)
    # Remove non-alphabetic characters
    clean_document = re.sub(r'[^a-zA-Z]', ' ', clean_document)
    # Remove stopwords
    clean_document = [token for token in clean_document.split() if token not in stopwords.words('english')]
    # Lemmatize words
    pos_tagged_tokens = nltk.pos_tag(clean_document)
    clean_document = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tagged_tokens]
    # Stem words
    # clean_document = [stemmer.stem(token) for token in clean_document]
    # Set labels
    label = row['cpc_subclass']
    # Add document and labels to model
    PLDA_model_IDF.add_doc(clean_document, labels=label)
    PLDA_model_ONE.add_doc(clean_document, labels=label)
    PLDA_model_PMI.add_doc(clean_document, labels=label)


    # Instantiate labelled LDA model (source: https://bab2min.github.io/tomotopy/v/en/#tomotopy.PLDAModel)
# Term weight options: IDF, ONE (equally weighted), PMI (Pointwise Mutual Information)

# Train a topic model using tomotopy library
PLDA_model_IDF.burn_in = 5
print('Start training model:')
for i in range(0, 100, 10):
    PLDA_model_IDF.train(iter=10, workers=0)
    # print('Iteration: {}\tLog-likelihood: {}'.format(i, PLDA_model.ll_per_word))

PLDA_model_ONE.burn_in = 5
print('Start training model:')
for i in range(0, 100, 10):
    PLDA_model_ONE.train(iter=10, workers=0)
    # print('Iteration: {}\tLog-likelihood: {}'.format(i, PLDA_model.ll_per_word))

PLDA_model_PMI.burn_in = 5
print('Start training model:')
for i in range(0, 100, 10):
    PLDA_model_PMI.train(iter=10, workers=0)
    # print('Iteration: {}\tLog-likelihood: {}'.format(i, PLDA_model.ll_per_word))

PLDA_model_IDF.summary()
PLDA_model_ONE.summary()
PLDA_model_PMI.summary()

# Save model
PLDA_model_IDF.save('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Tomotopy/PLDA_model_IDF.bin')
PLDA_model_ONE.save('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Tomotopy/PLDA_model_ONE.bin')
PLDA_model_PMI.save('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/Tomotopy/PLDA_model_PMI.bin')