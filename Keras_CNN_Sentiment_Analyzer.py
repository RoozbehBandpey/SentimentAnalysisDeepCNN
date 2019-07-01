import Data_Preprocess
from Word_2_Vector import train_word2vec
import numpy as np
import timeit
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

import matplotlib
import matplotlib.pyplot as plt

np.random.seed(2)



def build_CNN(X, Y, embedding_weights, sequence_length, embedding_dim, dropout_prob, hidden_dims, vocabulary ,num_filters = 3, filter_sizes = (3, 4), epoch_no = 100, batch_size = 32):
    network = Input(shape=(sequence_length, embedding_dim))

    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(network)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=network, output=out)


    model = Sequential()


    model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length, weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    result = model.fit(X, Y, batch_size=batch_size,  nb_epoch=epoch_no, validation_split=0.2, verbose=2)
    stop = timeit.default_timer()
    totaltime = stop - start
    plt.plot(result.history['acc'])
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_acc'])
    plt.plot(result.history['val_loss'])
    plt.title('Model type: '+"Embeddings dim: "+str(embedding_dim)+", Batch size: "+str(batch_size)
              +", Filters No.: "+str(num_filters))
    plt.ylabel('accuracy')
    plt.xlabel('epoch '+'( '+str(round(totaltime, 2))+' s )')
    plt.legend(['train', 'train err', 'test', 'test err'], loc='upper left')
    plt.figtext(.6,.2 , "Max: "+str(round(max(result.history['val_acc']), 4)*100)+" in epoch: "+str(result.history['val_acc'].index(max(result.history['val_acc']))+1))
    plt.show()





if __name__ == "__main__":
    start = timeit.default_timer()
    print("Loading data...")
    X, Y, vocabulary_inv, vocabulary = Data_Preprocess.load_data()
    print(X)
    print(Y)

    # model = "Baseline"
    model = "W2VCNN"

    sequence_length = len(X[0])
    min_word_count = 1
    context = 10
    batch_size = 32
    num_epochs = 100

    if model == "Baseline":
        embedding_dim = 20
        filter_sizes = (3, 4)
        num_filters = 150
        dropout_prob = (0.25, 0.5)
        hidden_dims = 150
        embedding_weights = None
    elif model == "W2VCNN":
        embedding_dim = 300
        filter_sizes = (3, 4)
        num_filters = 10
        dropout_prob = (0.7, 0.8)
        hidden_dims = 100
        embedding_weights = train_word2vec(X, vocabulary_inv, embedding_dim, min_word_count, context)
        print(len(embedding_weights[0]))



    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    x_shuffled = X[shuffle_indices]
    y_shuffled = Y[shuffle_indices]
    print(len(x_shuffled), len(y_shuffled))
    build_CNN(x_shuffled, y_shuffled,
              embedding_weights,sequence_length,
              embedding_dim, dropout_prob, hidden_dims,
              vocabulary, num_filters,
              filter_sizes, num_epochs, batch_size)