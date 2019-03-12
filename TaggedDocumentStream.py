from gensim.models.doc2vec import TaggedDocument
from itertools import islice

# A TaggedDocumentStream is an object that streams a combination of preprocessed notes and labels to
# the training procedure of the gensim Doc2Vec class.
class TaggedDocumentStream(object):

    # Initialize with a list of note files, a list of label files, and a number of maxrows
    def __init__(self, note_files, label_files, maxrows=None):
        self.note_files = note_files
        self.label_files = label_files
        self.maxrows = maxrows

    # Yield a TaggedDocument by iterating over the lines in a file
    def yield_td(self, note_file, label_file):

        # Keep track of row count
        row_counter = 0

        # Open note file and label file
        with open(note_file) as note_file, open(label_file) as label_file:

            # Iterate over lines
            for note, label in zip(note_file, label_file):

                # Check number of files that are read
                if row_counter == self.maxrows:
                    break
                row_counter += 1

                # Yield a TaggedDocument by splitting on whitespaces, and omitting the final newline character
                yield TaggedDocument(note[:-1].split(" "), [label[:-1]])

    # Implement iteration function by iterating over all note and label files
    def __iter__(self):

        for note_file, label_file in zip(self.note_files, self.label_files):
            yield from self.yield_td(note_file, label_file)