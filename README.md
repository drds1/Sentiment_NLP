# Sentiment Analysis: Tweets

These notebooks perform sentiment analysis on a series of tweets pertaining either to various disasters worldwide or mundane day-to-day benign tweets. The goal is to use Natural Language Processing to build a classifier model to tell them apart. The notebooks below trial 2 approaches:

1) [Naive Bayes Classifier](https://github.com/dstarkey23/disaster_nlp/blob/master/tweet_nbc.ipynb)
The Naive Bayes classifier consider occurence counts of words within each class and multiplies occurence probabilities of each word together assuming words are independent of one-another. This 'bag-of-words' assumption is modified slightly using the 'term frequency' and 'inverse document frequency' transformations to down-weight overly common words (e.g. 'the' and 'and'), and to normalise word counts for input samples of different word lengths. 

2) [Long-Short-Term Memory Neural Network](https://github.com/dstarkey23/disaster_nlp/blob/master/tweet_lstm.ipynb)
LSTMS have the advantage over Naive Bayes classifiers in that they are able to contextualise a word based on where it appears in a sequence (e.g. 'Not Bad' can be interpreted differently to 'Bad'). They offer an advancement over recurrent neural nets in that while both have memory of previous words in the sequence, LSTM's have a longer term memory and can pull context from many words ago. They are often referred to as 'fancy' RNN's.


The [benchmarking_models.py](https://github.com/dstarkey23/disaster_nlp/blob/master/benchmarking_models.py) script performs 3-fold cross validation on the two methods and uses an ROC AUC analysis to bechmark the performance of the classifiers against each other.

## Data Source:
The labelled tweet dataset is part of a [kaggle competition](https://www.kaggle.com/c/nlp-getting-started) with the objectives above.  

### Supplementary LSTM requirements
I use the GloVe non-contextual word embeddings to represent each word as a 100d vector.
The data set can be downloaded [here](http://nlp.stanford.edu/data/glove.6B.zip) or from a linux command line using
`wget http://nlp.stanford.edu/data/glove.6B.zip`.



## Summary:
It is expected the the LSTM outperforms the Naive Bayes Classifier. This appears to be the case from the ROC curve below based on 
10 epochs of training on a macbook pro. It should be noted at this point the LSTM appeared still unconverged. More training epochs recommended to exhaustively optimise the network.
Despite the undertraining, we see 10 epochs of LSTM training to be sufficient to outperform the Naive Bayes Classifier. Model performance is quantified by the area under the ROC curve below. A theoretical perfect classifier would return an area of 1.

![](https://github.com/dstarkey23/disaster_nlp/blob/master/ROC_curve_model_comparison.png)

A final word on interpreting the results here. Both our classifiers return the probability of a tweet belonging to the positive class. It is up to the user to determine the threshold probability at which to consider a tweet positive or negative. The ROC curve above trials several threshold probabilities (from 0.05 to 1 in 0.05 steps), recalculates the true positive and false positive rates and plots this as a point on the ROC curve. Traditionally, the threshold probability corresponding to the uppermost-left part of the curve is chosen as optimum (i.e the highest true positive but lowest false positive setting). In this case, the LSTM is expected to correctly identify 80pc of tweets, with a false positive rate of around 20pc. Further training epochs are expected to improve this subject to careful monitoring of training and test loss metrics to check for overfit.
