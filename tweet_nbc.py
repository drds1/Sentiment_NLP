import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import pickle
''' 1- Load the input data'''
df = pd.read_csv('./data/train.csv')
X = list(df['text'])
y = list(df['target'])

'''2 split into train and test samples'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


''' 3 setup a pipeline to 
a- compute the bag of words counts
b- apply tfidf trandormer to downweight common words and normalise long inputs
c- train the naive bayes classifier
'''
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])

'''pickle model output for benchmarking'''
picklefile = './models/naive_bayes.pickle'
os.system('rm ' + picklefile)
pickle_out = open(picklefile, "wb")
pickle.dump({'model':text_clf,
             'model_name':'Naive Bayes',
             'X_train':X_train,
             'y_train':y_train,
             'X_test':X_test,
             'y_test':y_test,
             'kwargs':{}}, pickle_out)
pickle_out.close()


'''
Optional following sections can be performed here or 
reserved for benchmarking with NB classifier
'''
perform_fit = True
if perform_fit:
    ''' Fit the model'''
    text_clf.fit(X_train, y_train)


    '''predict on test data'''
    y_pred = text_clf.predict(X_test)

    ''' 4 analyse performance '''
    y_pred = text_clf.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('ROC curve AUC = '+str(auc))


    #optional run K-fold cross validation to asses model performance
    #from benchmarking_models import run_cv
    #y_pred = run_cv(X_train, y_train)