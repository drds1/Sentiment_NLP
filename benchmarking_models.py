import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

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
text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)


''' 4 analyse performance '''
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print('ROC curve AUC = '+str(auc))
