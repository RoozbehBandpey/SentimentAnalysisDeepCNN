from terminaltables import AsciiTable
import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
from gensim.models import word2vec
from os.path import join, exists, split
import os
from nltk.tokenize import word_tokenize
from collections import Counter
import itertools
import time


def build_data_structure(input_files):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    train_file = input_files[0]
    test_file = input_files[1]
    vocab = defaultdict(float)
    with open(train_file, "r") as f:
        for line in f:
            rev = []
            temp_rev = []
            rev.append(line.strip())  # remove the "\n" at the end of each line
            temp_rev.append(rev[0].split('\t')[0])
            temp_rev.append(rev[0].split('\t')[1])

            orig_rev = clean_str(temp_rev[1])

            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            rev_data  = {"y":int(temp_rev[0]),
                        "text": orig_rev,
                        "num_words": len(orig_rev.split())}
            revs.append(rev_data)

    with open(test_file, "r") as f:
        for line in f:
            rev = []
            temp_rev = []
            rev.append(line.strip())  # remove the "\n" at the end of each line
            temp_rev.append(rev[0].split('\t')[0])
            temp_rev.append(rev[0].split('\t')[1])

            orig_rev = clean_str(temp_rev[1])

            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            rev_data  = {"y":int(temp_rev[0]),
                        "text": orig_rev,
                        "num_words": len(orig_rev.split())}
            revs.append(rev_data)

        splited = [s["text"].strip() for s in revs]
        tokenized_sentences = [word_tokenize(s) for s in splited]
        labels = [L["y"] for L in revs ]

    return revs, vocab, tokenized_sentences, labels





def review_data_structure(input_files):
    """
    Loads data and construct the proper data structure
    """
    revs = []
    file = input_files
    vocab = defaultdict(float)
    labels = []
    i = 1
    with open(file, "r") as f:
        rows_no = len(f.readlines())
        f.seek(0)

        for line in f:
            percent = float((i*100) / rows_no)
            hashes = '#' * int(round(percent/2))
            spaces = ' ' * (len(hashes)- 50)

            sys.stdout.write("\rLoading data: [{0}] {1}%".format(hashes + spaces, int(round(percent))))

            rev = []
            temp_rev = []
            rev.append(line.strip()) # remove the "\n" at the end of each line
            temp_rev.append(rev[0].split('\t')[0])
            temp_rev.append(rev[0].split('\t')[1])


            orig_rev = clean_str(temp_rev[1])

            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            labels.append(int(temp_rev[0]))
            # if temp_rev[0] == '0':
            #     labels.append([1,0])
            # elif temp_rev[0] == '1':
            #     labels.append([0,1])
            rev_data  = {"y":int(temp_rev[0]),
                         "text": orig_rev,
                         "num_words": len(orig_rev.split())}
            revs.append(rev_data)
            sys.stdout.flush()
            i += 1

    print("\t{0} loaded\n".format(f.name))
    # Split by words
    splited = [s["text"].strip() for s in revs]
    tokenized_sentences = [word_tokenize(s) for s in splited]


    return revs, vocab, tokenized_sentences, labels



def clean_str(string):
    """
    Tokenization/string cleaning/lower cased for datasets.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def rev_count(reviews):
    pos_rev = 0
    neg_rev = 0
    for item in reviews:
        if item["y"] == 0:
            neg_rev += 1
        else:
            pos_rev += 1

    return (pos_rev, neg_rev)


def sentence_padding(tokenized_sentences, max_length, padding_sign = "<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(tokenized_sentences)):
        percent = float((i * 100) / len(tokenized_sentences))
        hashes = '#' * int(round(percent / 2))
        spaces = ' ' * (len(hashes) - 50)

        sys.stdout.write("\rPadding sentences: [{0}] {1}%".format(hashes + spaces, int(round(percent))))

        sentence = tokenized_sentences[i]
        num_padding = max_length - len(sentence)
        new_sentence = sentence + [padding_sign] * num_padding
        padded_sentences.append(new_sentence)
        sys.stdout.flush()

    print("\tsentences are padded\n")

    return padded_sentences


def dictionary_maker(sentences):

    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return vocabulary, vocabulary_inv

def build_matrices(sentences, labels, vocabulary):
    """
        Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return x, y


def load_data():
    input_files = ["Input_Files\\train.txt", "Input_Files\\test.txt"]

    """
    revs: A list of dictionaries in following format: [{'text': 'review', 'y': label, 'num_words': number of words in review},...]
    vocab: A dictionary in following format: {'word': frequency,...}
    tokenized_sentences: A 2D list in following format: [[tokenized_sentences_1],...]
    labels: A list with represent label of reviews respectively
    """
    # train_revs, train_vocab, train_tokenized_sentences, train_labels = review_data_structure(input_files[0])
    # test_revs, test_vocab, test_tokenized_sentences, test_labels = review_data_structure(input_files[1])

    # train_max_sentence_lenght = np.max(pd.DataFrame(train_revs)["num_words"])  # Get the max length sentence
    # test_max_sentence_lenght = np.max(pd.DataFrame(test_revs)["num_words"])  # Get the max length sentence

    revs, vocab, tokenized_sentences, labels = build_data_structure(input_files)

    max_sentence_lenght = np.max(pd.DataFrame(revs)["num_words"])  # Get the max length sentence

    table_data = []
    table_data.append(['Data Statistics', 'Values'])
    table_data.append(['Number of sentences', str(len(revs))])
    table_data.append(['Vocabulary size', str(len(vocab))])
    table_data.append(['Max sentence length', str(max_sentence_lenght)])
    table_data.append(['Positive reviews', str(rev_count(revs)[0])])
    table_data.append(['Negative reviews', str(rev_count(revs)[1])])
    train_table = AsciiTable(table_data)
    print(train_table.table)


    # train_table_data = []
    # train_table_data.append(['Training Data Statistics', 'Values'])
    # train_table_data.append(['Number of sentences', str(len(train_revs))])
    # train_table_data.append(['Vocabulary size', str(len(train_vocab))])
    # train_table_data.append(['Max sentence length', str(train_max_sentence_lenght)])
    # train_table_data.append(['Positive reviews', str(rev_count(train_revs)[0])])
    # train_table_data.append(['Negative reviews', str(rev_count(train_revs)[1])])
    # train_table = AsciiTable(train_table_data)
    # print(train_table.table)
    #
    # test_table_data = []
    # test_table_data.append(['Test Data Statistics', 'Values'])
    # test_table_data.append(['Number of sentences', str(len(test_revs))])
    # test_table_data.append(['Vocabulary size', str(len(test_vocab))])
    # test_table_data.append(['Max sentence length', str(test_max_sentence_lenght)])
    # test_table_data.append(['Positive reviews', str(rev_count(test_revs)[0])])
    # test_table_data.append(['Negative reviews', str(rev_count(test_revs)[1])])
    # test_table = AsciiTable(test_table_data)
    # print(test_table.table)

    """
    sentences_padded: A 2D list in following format: [[tokenized_sentences_1+paddings],...], max value: 3256, min value: 0
    """
    # train_sentences_padded = sentence_padding(train_tokenized_sentences, train_max_sentence_lenght)
    # test_sentences_padded = sentence_padding(test_tokenized_sentences, test_max_sentence_lenght)
    #
    sentences_padded = sentence_padding(tokenized_sentences, max_sentence_lenght)
    #
    # train_vocabulary, train_vocabulary_inv = dictionary_maker(train_sentences_padded)
    # test_vocabulary, test_vocabulary_inv = dictionary_maker(test_sentences_padded)
    #
    vocabulary, vocabulary_inv = dictionary_maker(sentences_padded)
    #
    # X_train, y_train = build_matrices(train_sentences_padded, train_labels, train_vocabulary)
    # X_test, y_test = build_matrices(test_sentences_padded, test_labels, test_vocabulary)
    #
    X, Y = build_matrices(sentences_padded, labels, vocabulary)
    #
    # return (X_train, y_train, train_vocabulary, train_vocabulary_inv), (X_test, y_test, test_vocabulary, test_vocabulary_inv)
    return X, Y, vocabulary_inv, vocabulary


if __name__ =="__main__":

    X, Y, vocabulary_inv, vocabulary = load_data()
    print(len(X))
    print(len(Y))
    print(len(vocabulary_inv))
    print(len(vocabulary))