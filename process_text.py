import numpy as np

import nltk.data
from nltk.tokenize import word_tokenize

import unidecode
import re

# Define tokenization procedure
sent_tokenizer = nltk.data.load("file:models/dutch.pickle")

def tokenize(text):
    for sentence in sent_tokenizer.tokenize(text):
        yield word_tokenize(sentence)

# Read stopwords
with open('models/dutch_stopwords') as f:
    dutch_stopwords = set(f.read().splitlines())

# Initialize stemmer (using package nltk)
stemmer = nltk.stem.snowball.DutchStemmer()

# Preprocessing for text
def text_to_words(text, filter_stopwords=True, stemming=False, filter_periods=False):

    # Lowercase and remove special characters (ë => e, etc)
    text = text.lower()
    text = unidecode.unidecode(text)

    # Remove all non space, period, lowercase
    text = re.sub(r'([^\sa-z\.]|_)+', ' ', text)

    # Remove obsolete periods
    text = re.sub(r'\s\.\s', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r' +', ' ', text)

    # Tokenize
    words = [word for sentence in tokenize(text) for word in sentence]

    # Filter stopwords
    if filter_stopwords:
        words = [word for word in words if word not in dutch_stopwords]

    # Stemming
    if stemming:
        words = [stemmer.stem(w) for w in words]

    # Filter periods
    if filter_periods:
        words = [word for word in words if word != "."]

    # Return
    return words

# Convert a dataframe with texts in the 'text_column' column to a numpy array with vector representations,
# based on a paragraph2vec_model and a specified number of repetitions.
def text_to_vectors(notes_df, text_column, paragraph2vec_model, no_reps=10):

    # Output is a matrix with rows equal to number of notes, and columns equal to paragraph2vec model size
    note_vectors = np.zeros((len(notes_df), paragraph2vec_model.vector_size))

    # Iterate over all notes
    for i in notes_df.index:

        # Words are in the 'text_preprocessed' column split by whitespaces
        note_words = notes_df.loc[i, text_column].split(" ")

        # Initialize an empty vector of length paragraph2vec model size
        note_vec = np.zeros((paragraph2vec_model.vector_size))

        # Iterate over number of repetitions to cancel out inaccuracies
        for _ in range(no_reps):
            note_vec += paragraph2vec_model.infer_vector(note_words)

        # Add to note_vectors after normalizing for number of repetitions
        note_vectors[i] = (note_vec / no_reps)

    # Return output
    return note_vectors