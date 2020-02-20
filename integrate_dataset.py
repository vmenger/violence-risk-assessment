# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:53:19 2020

@author: Moste007
"""
import gensim.models as gm
import pandas as pd
import process_text as pt
import csv
import os

def read_data():
    # Read admissions
    admissions = pd.read_csv("data/source/admissions.csv", 
                             sep=";",
                             parse_dates=['start_datetime', 'end_datetime']
                             )
    
    # Read incidents
    incidents = pd.read_csv("data/source/incidents.csv", 
                            sep=";",
                            parse_dates=['datetime']
                            )
    
    # Read notes
    notes = pd.read_csv("data/source/notes.csv", 
                        sep=";", 
                        parse_dates=['datetime']
                        )
    
    # Read trained paragraph2vec model
    paragraph2vec_model = gm.Doc2Vec.load('models/paragraph2vec_model')
    return admissions, incidents, notes, paragraph2vec_model

def process_incidents(admissions, incidents):
    # Inner join admissions and incidents
    adm_incidents = admissions[['patient_id', 'start_datetime', 'admission_id']].merge(
        incidents[['patient_id', 'datetime']], how='inner')
    
    # Determine how much time between start of admission and each incident
    adm_incidents['day_diff'] = (adm_incidents['datetime'] - adm_incidents['start_datetime']) 
    adm_incidents['day_diff'] = adm_incidents['day_diff'] / pd.Timedelta("24 hour")
    
    # Only retain incidents after the first 24 hours, and up to the first 28 days of admission
    adm_incidents = adm_incidents[(adm_incidents['day_diff'] >= 1) & (adm_incidents['day_diff'] <= 28)]
    
    # Group incidents for each admission, by simply taking the first if multiple are present
    adm_incidents = adm_incidents.groupby("admission_id").first()
    adm_incidents = adm_incidents.drop_duplicates()
    adm_incidents = adm_incidents.reset_index()
    
    # Merge this dataframe back to the original 
    admissions = admissions.merge(adm_incidents[['admission_id', 'day_diff']], how='left')
    
    # Determine outcome (i.e. the day_diff variable is not empty)
    admissions['outcome'] = admissions['day_diff'].notnull()
    admissions['outcome'] = admissions['outcome'].map({False : 0, True : 1})
    return admissions
    
def process_notes(notes, admissions, paragraph2vec_model):
    # Inner join admission info
    notes = notes.merge(
        admissions[['patient_id', 'admission_id', 'start_datetime', 'transfer', 'outcome']],
        how='inner',
        left_on='patient_id', 
        right_on='patient_id'
    )
    
    # Determine how much time between start of admission and each note
    notes['day_diff'] = (notes['start_datetime'] - notes['datetime']) 
    notes['day_diff'] = notes['day_diff'] / pd.Timedelta("24 hour")
    
    # Determine a threshold for inclusion of retrospective notes (i.e. one week of four weeks)
    notes['threshold'] = notes['transfer'].apply(lambda x : 7 if x else 28)
    
    # Retain notes that are after the threshold, and before 24 hours have passed
    notes = notes[(notes['day_diff'] <= notes['threshold'])]
    notes = notes[(notes['day_diff'] > -1)]
    
    # Concatenate multiple notes into a single text, add a newline character in between
    notes_concat = notes.groupby("admission_id")['text'].agg(lambda x : "\n".join(x)).reset_index() # add
    
    # Omit notes with fewer than 100 words
    notes_concat['no_words'] = notes_concat['text'].apply(lambda x : len(x.split(" "))) 
    notes_concat = notes_concat[notes_concat['no_words'] > 100]
    
    # Convert text to words
    notes_concat['words_stemmed'] = notes_concat['text'].apply(lambda x : pt.text_to_words(x, 
                                                                                        filter_stopwords=True,
                                                                                        stemming=True,
                                                                                        filter_periods=True
                                                                                        ))
    
    # Join with whitespace
    notes_concat['words_stemmed'] = notes_concat['words_stemmed'].apply(lambda x : ' '.join(x))
    notes_concat = notes_concat.reset_index()
    
    # Convert text to notes
    note_vectors = pt.text_to_vectors(notes_concat, 'words_stemmed', paragraph2vec_model, 10)
    
    # Concatenate to original dataframe
    notes_concat = pd.concat([notes_concat, pd.DataFrame(note_vectors)], axis=1)
    
    # Merge outcome from admission table
    notes_concat = notes_concat.merge(admissions[['outcome', 'admission_id', 'patient_id']])
    
    # Write processed data to file for other notebooks
    processed_dir = os.path.join("data", "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    notes_concat.to_csv(os.path.join(processed_dir, "notes.csv"), 
                        sep=";", 
                        index=False, 
                        quoting=csv.QUOTE_ALL)

def integrate_dataset():
    admissions, incidents, notes, paragraph2vec_model = read_data()
    processed_admissions = process_incidents(admissions, incidents)
    process_notes(notes, processed_admissions, paragraph2vec_model)
    
    
if __name__ == "__main__":
    integrate_dataset()
    
