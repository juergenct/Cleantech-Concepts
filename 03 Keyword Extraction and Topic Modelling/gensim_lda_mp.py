import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

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
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)).lower() for w in tokens if w.isalpha() and w.lower() not in stop_words]
    return tokens

# Initialize dictionary and corpus
dictionary = corpora.Dictionary()
corpus = []

# Function to process a chunk of data
def process_chunk(chunk):
    global dictionary, corpus
    texts = chunk['text'].map(preprocess_text)
    dictionary.add_documents(texts)
    corpus_chunk = [dictionary.doc2bow(text) for text in texts]
    corpus.extend(corpus_chunk)

# List of file paths
file_paths = [
    '/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LDA/df_rel_cleantech.json',
    '/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LDA/df_uspto_cleantech.json',
    '/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LDA/df_epo_cleantech.json'
]

# Process each file
chunksize = 10000  # Adjust based on your memory availability
for file_path in file_paths:
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        process_chunk(chunk)

# Train the LDA model using LdaMulticore
lda_model = models.LdaMulticore(corpus, num_topics=10, id2word=dictionary, passes=10, workers=3)

# Print the topics
for i in range(10):  # Assuming you have 10 topics
    words = lda_model.show_topic(i)
    print(f"Topic {i}:")
    print(", ".join([word for word, prob in words]))

# Save the model (optional)
lda_model.save('/mnt/hdd01/patentsview/Patentsview - Cleantech Patents/Cleantech Concepts/LDA/full_lda_model.model')
