import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models import Phrases
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

# # Download necessary NLTK data
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# Function to get the correct POS tag for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess and lemmatize text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)).lower() for w in tokens if w.isalpha() and w.lower() not in stop_words]

# Function to process data with multiprocessing
def process_data_parallel(data_texts):
    with Pool(min(multiprocessing.cpu_count(), 12)) as pool:
        processed_texts = list(tqdm(pool.imap(preprocess_text, data_texts), total=len(data_texts)))
    return processed_texts

# Path to the data file
file_path = '/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LSA/df_epo_uspto_rel_cleantech.json'

# Process the data
data = pd.read_json(file_path)
data = data[data['text'].apply(lambda x: isinstance(x, str))]
data = data.reset_index(drop=True)
# data = data.sample(n=500000, random_state=42)

# Process the data and update the dictionary
processed_texts = process_data_parallel(data['text'])

# Compute bigrams
bigram = Phrases(processed_texts, min_count=20)
for idx in range(len(processed_texts)):
    for token in bigram[processed_texts[idx]]:
        if '_' in token:
            # Token is a bigram, add to document
            processed_texts[idx].append(token)

# Create a dictionary representation of the documents and filter extremes
dictionary = corpora.Dictionary(processed_texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create a streaming corpus
class MyCorpus(object):
    def __init__(self, texts, dictionary):
        self.texts = texts
        self.dictionary = dictionary

    def __iter__(self):
        for text in self.texts:
            yield self.dictionary.doc2bow(text)

corpus = MyCorpus(processed_texts, dictionary)

# Perform Latent Semantic Analysis
lsa_model = models.LsiModel(corpus, num_topics=10, id2word=dictionary)

# # Print the topics
# for i in range(10):  # Assuming you have 10 topics
#     words = lsa_model.show_topic(i)
#     print(f"Topic {i}:")
#     print(", ".join([word for word, prob in words]))

# Save the model (optional)
lsa_model.save('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LSA/full_lsa_model.model')
