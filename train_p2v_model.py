# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:40:37 2020

@author: Moste007
"""
import pandas as pd
import nltk
nltk.download('punkt')
import process_text as pt
import os
import TaggedDocumentStream as tds
import gensim.models as gm

def read_notes(notes_filename):
    return pd.read_csv(notes_filename, sep=";")

def write_outfiles(notes, notes_file, label_file):
    # Iterate over records (== notes)
    for i in notes.index:

        # Extract text and label
        text = notes.loc[i]['text']
        label = notes.loc[i]['label']

        # Convert text to words
        words = pt.text_to_words(text, 
                                 filter_stopwords=True,
                                 stemming=True,
                                 filter_periods=True
                                 )

        # Only texts with at least 2 words
        if len(words) <= 1:
            continue

        # Append to file 
        notes_file.write("{}\n".format(' '.join(words)))
        label_file.write("{}\n".format(label))

def train_p2v_model(notes_infilename, notes_outfilename, labels_outfilename):
    notes = read_notes(notes_infilename)
    # Open file handles for preprocessed notes and lables
    with open(notes_outfilename, 'a+') as notes_file, open(labels_outfilename, 'a+') as label_file:
        write_outfiles(notes, notes_file, label_file)
    # Define TaggedDocumentStream
    notes_stream = tds.TaggedDocumentStream(note_files=[notes_file.name], label_files=[label_file.name])
    # Train paragraph2vec model
    paragraph2vec_model = gm.Doc2Vec(documents = notes_stream, 
                                     epochs=20,
                                     min_count=20,
                                     dm=1,
                                     sample=1e-3,
                                     vector_size=300, 
                                     window=2)
    # Save model to disk
    paragraph2vec_model.save("models/paragraph2vec_model")

if __name__ == "__main__":
    notes_filename = "data/source/notes_full.csv"

    # Processed notes and labels are written to two seperate files
    notes_file_path = os.path.join('data', 'processed_notes', 'notes.txt')
    label_file_path = os.path.join('data', 'processed_notes', 'labels.txt')

    train_p2v_model(notes_filename, notes_file_path, label_file_path)
