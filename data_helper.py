from terminaltables import AsciiTable
import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd

def cross_vlaidation_data(input_files, fold = 10, clean_string = True):
    """
    Loads data and construct the proper data structure
    """
    revs = []
    pos_file = input_files[0]
    neg_file = input_files[1]
    vocab = defaultdict(float)
    with open(pos_file, "r") as f:
        for line in f:
            rev = []
            rev.append(line.strip()) # remove the "\n" at the end of each line

            if clean_string:
                orig_rev = clean_str(rev[0])
                # print(orig_rev)
            else:
                orig_rev = rev[0].lower()
                print(orig_rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            date  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,fold)}
            revs.append(date)

    with open(neg_file, "r") as f:
        for line in f:
            rev = []
            rev.append(line.strip())

            if clean_string:
                orig_rev = clean_str(rev[0])
            else:
                orig_rev = rev[0].lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            date  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,fold)}
            revs.append(date)
    return revs, vocab

def get_W(word_vecs, k = 300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """

    print("loading word2vec vectors please wait...")
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size

        i = 1
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

            sys.stdout.write("\r%d%%" % i)
            sys.stdout.flush()
            i += 100 / vocab_size

    print('\n')
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
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
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence]
                  for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(
        open("Input_Files\\positive-polarity.txt").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open("Input_Files\\negative-polarity.txt").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


if __name__=="__main__":
    w2v_file = "Input_Files\\GoogleNews-vectors-negative300.bin"
    input_files = ["Input_Files\\positive-polarity.txt", "Input_Files\\negative-polarity.txt"]
    print("loading data...")
    revs, vocab = cross_vlaidation_data(input_files, fold = 10, clean_string = True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])#Get the max length sentence

    print("data loaded!")

    table_data = []
    table_data.append(['Data Statistics', 'Val'])
    table_data.append(['Number of sentences', str(len(revs))])
    table_data.append(['Vocabulary size', str(len(vocab))])
    table_data.append(['Max sentence length', str(max_l)])
    table = AsciiTable(table_data)
    print(table.table)

    print("word2vec vectors loaded: ", load_bin_vec(w2v_file, vocab))
    # print("word2vec loaded!")
    # print("num words already in word2vec: " + str(len(w2v)))
    # add_unknown_words(w2v, vocab)
    # W, word_idx_map = get_W(w2v)
    # rand_vecs = {}
    # add_unknown_words(rand_vecs, vocab)
    # W2, _ = get_W(rand_vecs)
    # cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    # print("dataset created!")
