# Sentiment Classification with Convolutional Neural Networks
## Research Paper

[Sentiment Classification with Convolutional Neural Networks - Paper](https://www.researchgate.net/publication/341322592_Sentiment_Classification_with_Convolutional_Neural_Networks)
[Sentiment Classification with Convolutional Neural Networks - Poster](https://www.researchgate.net/publication/341322681_Sentiment_Classification_with_Deep_Convolutional_Neural_Networks)
_____________________________________________________________________
## Overview
Sentiment Analysis of amazon product reviews with deep convolutional neural network.
The main focus of this work is a model that initializes the parameter weights of the convolutional neural network, which is effective to train an accurate model even for a small data-set. In a nutshell, we use a model to train initial word embeddings that are further tuned by our deep learning model. At a final stage, the pre-trained parameters of the network are used to initialize the model. Later on we compared this model with a baseline model, which takes randomly initialized word vectors as input.

## Required libraries:
[Requirments](./requirements.txt)

## Run
Simply run [Keras_CNN_Sentiment_Analyzer](./Keras_CNN_Sentiment_Analyzer.py)

## Data
We ran our experiments on a data-set which is customer reviews of various products, sentences are annotated as positive or negative sentiment. We randomly  select  20%  of  the  training  data  as  the development set. For extracting vectors out of words and further on constructing matrices for sentences, we use google word2vec that were trained on 100 billion words from Google News (Mikolov, 2013). For this pur-pose  we  used  a  python  library  called gensim.This library practically invokes the trained google model and it is possible to use it on any vocabulary. 