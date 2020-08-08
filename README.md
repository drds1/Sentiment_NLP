# Sentiment Analysis: Tweets

These notebooks perform sentiment analysis on a series of tweets pertaining either to various disasters worldwide or mundane day-to-day benign tweets. The goal is to train a model to tell them apart. The notebooks below trial 2 approaches:

1) Naive Bayes Classifier
The Naive Bayes classifier consider occurence counts of words within each class and multiplies occurence probabilities of each word together assuming words are independent of one-another. This 'bag-of-words' assumption is modified slightly using the 'term frequency' and 'inverse document frequency' transformations to down-weight overly common words (e.g. 'the' and 'and'), and to normalise word counts for input samples of different word lengths. 

2) Long-Short-Term Memory Neural Network
LSTMS have the advantage over Naive Bayes classifiers in that they are able to contextualise a word based on where it appears in a sequence (e.g. 'Not Bad' can be interpreted differently to 'Bad'). They offer an advancement over recurrent neural nets in that while both have memory of previous words in the sequence, LSTM's have a longer term memory and can pull context from many words ago. They are often referred to as 'fancy' RNN's.


The `benchmarking_models.py` script performs 4-fold cross validation on the two methods and uses an ROC AUC analysis to bechmark the performance of the classifiers against each other.

## Data Source:
The labelled tweet dataset is part of a [kaggle competition](https://www.kaggle.com/c/nlp-getting-started) with the objectives above.  

### Supplementary LSTM requirements
I use the GloVe non-contextual word embeddings to represent each word as a 100d vector.
The data set can be downloaded [here](http://nlp.stanford.edu/data/glove.6B.zip) or from a linux command line using
`wget http://nlp.stanford.edu/data/glove.6B.zip`.



## Summary:
It is expected the the LSTM outperforms the Naive Bayes Classifier. This appears to be the case based on 
10 epochs of training on a macbook pro. It should be noted at this point the LSTM appeared still unconverged. Larger machines recommended to exhaustively optimise the network.
Despite the undertraining, we see 10 epochs of LSTM training to be sufficient to outperform the Naive Bayes Classifier. Model performance is quantified by the area under the ROC curve below. A theoretical perfect classifier would return an area of 1.

![]('https://github.com/dstarkey23/disaster_nlp/blob/master/ROC_curve_model_comparison.png')



