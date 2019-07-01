Institutions: Universität Stuttgart
Goal: I am trying to work on Sentiment Analysis of Twitter posts.
The goal is to achieve higher accuracy by detecting the sentiment of sarcastic sentences.
My hypothesis for performing the sarcasm detection is that most of the sarcastic sentences must follow unique pattern.
Effectively solving this task requires strategies that combine the small text content with prior knowledge and use
more than just bag-of-words.
I assume deep Convolutional Neural Network that can exploits from character- to sentence-level information to perform
sentiment analysis of short text, can gain higher accuracy compare to traditional ways (non-deep learning).

My supervisor is Dr. Thang Vu, Departement of Natural Language Processing, University of Stuttgart.
Date: 10 December 2016 - 9 February 2017
Methods: Pattern Recognition, Deep Learning, Sentiment Analysis


As an interface to word2vec, we decided to go with a Python package called gensim.
gensim appears to be a popular NLP package, and has some nice documentation and tutorials, including for word2vec.
Data_Preprocess.py provides the proper data for Word_2_Vector.py and there we train our word2vector with gensim

Another option is use Google's pretarined word2vector (recommended).
It includes word vectors for a vocabulary of 3 million words and phrases
that they trained on roughly 100 billion words from a Google News dataset. The vector length is 300 features.

Loading this model using gensim is a piece of cake; you just need to pass in the path to the model file
(update the path in the code below to wherever you’ve placed the file).
###############################################################################################################
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
###############################################################################################################

memory error! If python 32bit is running
because gensim allocates a big matrix to hold all of the word vectors

3 million words * 300 features * 4bytes/feature = ~3.35GB

Here are some the questions I had about the vocabulary, which I answered in this project:

Does it include stop words?
Answer: Some stop words like “a”, “and”, “of” are excluded, but others like “the”, “also”, “should” are included.

Does it include misspellings of words?
Answer: Yes. For instance, it includes both “mispelled” and “misspelled”–the latter is the correct one.

Does it include commonly paired words?
Answer: Yes. For instance, it includes “Soviet_Union” and “New_York”.

Does it include numbers?
Answer: Not directly; e.g., you won’t find “100”.
But it does include entries like “###MHz_DDR2_SDRAM” where I’m assuming the ‘#’ are intended to match any digit.

