import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

# Download necessary NLTK data
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
    with Pool(12) as pool:  # Using xx processes
        processed_texts = list(tqdm(pool.imap(preprocess_text, data_texts), total=len(data_texts)))
    return processed_texts

# Initialize dictionary and corpus
dictionary = corpora.Dictionary()
corpus = []

# Path to the data file
file_path = '/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LSA/df_epo_uspto_rel_cleantech.json'

# Process the data
data = pd.read_json(file_path)
# Delete all rows of column 'text' that are not strings
data = data[data['text'].apply(lambda x: isinstance(x, str))]
# Reset index
data = data.reset_index(drop=True)
# Randomly sample x rows
data = data.sample(n=250000, random_state=42)
# Process the data
processed_texts = process_data_parallel(data['text'])

# Update dictionary and create corpus
for text in processed_texts:
    dictionary.add_documents([text])
    corpus_data = dictionary.doc2bow(text)
    corpus.append(corpus_data)

# Perform Latent Semantic Analysis
lsa_model = models.LsiModel(corpus, num_topics=10, id2word=dictionary)

# # Print the topics
# for i in range(10):  # Assuming you have 10 topics
#     words = lsa_model.show_topic(i)
#     print(f"Topic {i}:")
#     print(", ".join([word for word, prob in words]))

# Save the model (optional)
lsa_model.save('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LSA/full_lsa_model.model')
