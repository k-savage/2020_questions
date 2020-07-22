import nltk
import numpy as np
import string
import sys
import os
import glob
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data_folder = os.path.join(directory, "*")
    result = {}
    for filename in glob.glob(data_folder):
        with open(filename, 'r') as file:
            result[os.path.basename(filename)] = file.read()
    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    #format sentence to lowercast
    sentence = word_tokenize(document.lower())
    lst = []
    for item in sentence:
        #confirm word has at least one alpha numeric character
        if any(c.isalpha() for c in item):
            item = item.translate(str.maketrans('', '', string.punctuation))
            #confirm item not in stopwords
            if item not in nltk.corpus.stopwords.words("english"):
                lst.append(item)
    return lst


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    #calculate numerator
    total_documents = len(documents.keys())
    #create dictionary of counted words
    idfs_dict = sum((Counter(set(x)) for x in documents.values()), Counter())
    for word in idfs_dict:
        #perform IDF calculation
        idfs_dict[word] = np.log(total_documents / idfs_dict[word])
    return idfs_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words),
    `files` (a dictionary mapping names of files to a list of their words),
    and `idfs` (a dictionary mapping words to their IDF values),
    return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    #create empty dictionary with integer values
    dict_file_tfidf = defaultdict(int)
    for word in query:
        for doc_name, list_of_words in files.items():
            #Add TFIDF values per document
            dict_file_tfidf[doc_name] += list_of_words.count(word) * idfs[word]
    result = sorted(dict_file_tfidf, key=dict_file_tfidf.__getitem__, reverse=True)
    return result[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # create empty dictionary with integer values
    dict_sentence_scores = defaultdict(int)
    for item in query:
        for sentence, list_of_words in sentences.items():
            #create set of words already accounted for, to ignore term frequency
            set_to_remove = set()
            for word in list_of_words:
                if word == item and word not in set_to_remove:
                    #Sum IDFs for words in doc
                    dict_sentence_scores[sentence] += idfs[item]
                    #remove word from for loop to ignore term frequency
                    set_to_remove.add(word)
    result = sorted(dict_sentence_scores, key=dict_sentence_scores.__getitem__, reverse=True)
    return result[:n]


if __name__ == "__main__":
    main()
