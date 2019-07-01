from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

import Data_Preprocess


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = 'word2vec_models'
    model_name = "{0}features_{1}minwords_{2}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print("Training Word2Vec model...")
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                                            size=num_features, min_count=min_word_count, \
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model \
                                       else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) \
                                   for w in vocabulary_inv])]

    return embedding_weights

if __name__=='__main__':
    X, Y, vocabulary_inv, vocabulary = Data_Preprocess.load_data()
    #(X_train, y_train, train_vocabulary, train_vocabulary_inv), (X_test, y_test, train_vocabulary, train_vocabulary_inv)
    # X_train = data[0][0]
    # y_train = data[0][1]
    # train_vocabulary = data[0][2]
    # train_vocabulary_inv = data[0][3]
    # X_test = data[1][0]
    # y_test = data[1][1]
    # test_vocabulary = data[1][2]
    # test_vocabulary_inv = data[1][3]
    #
    #
    # train_word_embeddings = train_word2vec("train",X_train, train_vocabulary_inv)

    word_embeddings = train_word2vec(X, vocabulary_inv)
    print(len(word_embeddings[0]))
    # print(word_embeddings[0][0])
    # print(vocabulary_inv[0])
    # test_word_embeddings = train_word2vec("test",X_test, test_vocabulary_inv)
    # print(len(w))
    # print(len(w[0]))
    # print(len(w[0][0]))
