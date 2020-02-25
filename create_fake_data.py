# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:39:12 2020

@author: Moste007
"""

'''
The input will be a list of notes.
We filter out those that are under 100 words.
We make sure there are at least 1000 notes, otherwise we draw random words
We then create all the necessary files
'''
import random as r
import pandas as pd
import sys

def remove_short_notes(notes, min_note_length):
    return [el for el in notes if len(el.split(' ')) > min_note_length]

def get_number_of_missing_notes(notes, min_number):
    if len(notes) > min_number:
        return 0
    return min_number - len(notes)

def create_missing_notes(vocabulary, number_of_missing_notes, note_length):
    new_notes = []
    for _ in range(number_of_missing_notes):
        note = ""
        for _ in range(note_length):
            note += vocabulary[r.randint(0, len(vocabulary) - 1)]
            note += " "
        new_notes.append(note)
    return new_notes

def create_vocabulary(notes):
    vocabulary = []
    for note in notes:
        vocabulary += note.split(' ')
    return vocabulary

def create_output_files(notes):
    num_patients = len(notes)
    num_incidents = int(num_patients / 2)
    create_admissions(num_patients)
    create_incidents(num_patients, num_incidents)
    create_notes(num_patients, notes)
    create_notes_full(notes)
    
def create_admissions(num_patients):
    columns = ["patient_id", "admission_id", "start_datetime", "end_datetime", "transfer"]
    df = pd.DataFrame(columns = columns)
    df["patient_id"] = [i for i in range(num_patients)]
    df["admission_id"] = [i for i in range(num_patients)]
    df["start_datetime"] = ["2020-01-01 12:00" for _ in range(num_patients)]
    end_datetimes = []
    for _ in range(num_patients):
        day = r.randint(10, 20)
        datetime = "2020-01-" + str(day) + " 12:00"
        end_datetimes.append(datetime)
    df["end_datetime"] = end_datetimes
    df["transfer"] = [False for _ in range(num_patients)]
    df.to_csv("data/source/admissions.csv", sep=";", index = False)
    
def create_incidents(num_patients, num_incidents):
    if num_incidents > num_patients:
        num_incidents = num_patients
    columns = ["patient_id", "incident_id", "datetime"]
    df = pd.DataFrame(columns = columns)
    df["patient_id"] = [i for i in range(num_incidents)]
    df["incident_id"] = [i for i in range(num_incidents)]
    datetimes = []
    for _ in range(num_incidents):
        day = r.randint(1, 9)
        hour = r.randint(13, 23)
        minute = r.randint(10, 59)
        datetime = "2020-01-0" + str(day) + " " + str(hour) + ":" + str(minute)
        datetimes.append(datetime)
    df["datetime"] = datetimes
    df.to_csv("data/source/incidents.csv", sep=";", index = False)
    
def create_notes(num_patients, notes):
    if num_patients > len(notes):
        num_patients = len(notes)
    columns = ["patient_id", "note_id", "datetime", "text"]
    df = pd.DataFrame(columns = columns)
    df["patient_id"] = [i for i in range(num_patients)]
    df["note_id"] = [i for i in range(num_patients)]
    df["datetime"] = ["2020-01-01 13:00" for _ in range(num_patients)]
    df["text"] = notes[:num_patients]
    df.to_csv("data/source/notes.csv", sep=";", index = False)
    
def create_notes_full(notes):
    columns = ["label", "text"]
    df = pd.DataFrame(columns = columns)
    df["label"] = [i for i in range(len(notes))]
    df["text"] = notes
    df.to_csv("data/source/notes_full.csv", sep=";", index = False)
    
def read_notes(notes_file):
    with open(notes_file, encoding='latin-1') as f:
        return f.readlines()
    
def full_program(notes_file, min_num_notes, min_note_length):
    raw_notes = read_notes(notes_file)
    long_notes = remove_short_notes(raw_notes, min_note_length)
    num_missing_notes = get_number_of_missing_notes(long_notes, min_num_notes)
    vocab = create_vocabulary(raw_notes)
    new_notes = create_missing_notes(vocab, num_missing_notes, min_note_length)
    notes = long_notes + new_notes
    print(len(notes))
    print(notes[0])
    create_output_files(notes)
    
if __name__ == "__main__":
    inputs = sys.argv
    if len(inputs) != 3:
        print("Inputs: <notes_file> <min_num_notes>")
        exit(-1)
    full_program(inputs[1], int(inputs[2]), 100)
