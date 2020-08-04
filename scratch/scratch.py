import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pylab as plt
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def count_kw(dat,column = 'keyword'):

    kwnon = dat[(dat[column] != dat[column])]['target'].value_counts()
    kwnon.index = ['no '+column+' '+ str(i) for i in kwnon.index]
    kwinc = dat[(dat[column] == dat[column])]['target'].value_counts()
    kwinc.index = ['yes ' + column + ' ' + str(i) for i in kwinc.index]
    return kwnon.append(kwinc)

def keyword_study(data):
    '''

    :param kw:
    :param file:
    :return:
    '''
    countloc = count_kw(data,column = 'location')
    countkw = count_kw(data,column='keyword')
    counts = countloc.append(countkw)
    return counts

class nlp_model:

    def __init__(self):
        self.train_file = './data/train.csv'
        self.test_file = './data/test.csv'
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            #('clf',MultinomialNB()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, random_state=42,
                                  max_iter=5, tol=None)),
        ])


    def load_data(self,verbose = True):
        self.train = pd.read_csv(self.train_file)
        self.test  = pd.read_csv(self.test_file)

        if verbose is True:
            '''some exploratory analysis'''
            # identify common key words
            keywords = self.test['keyword'].value_counts(dropna=False)
            print('most common keywords...\n', keywords.head(10))
            print('\nleast common keywords...\n', keywords.tail(10))
            print('\n\n\n')
            # identify common locations
            locations = self.test['location'].value_counts(dropna=False)
            print('most common locations...\n', locations.head(10))
            print('\nleast common locations...\n', locations.tail(10))
            print('\n\n\n')
            #differences between keyword / not keyword
            print('keyword/keyloc breakdown...')
            print(keyword_study(self.train))


    def assemble_X_y(self,add_loc = True,add_keyword = True):
        '''

        :return:
        '''
        combined = self.train['text']

        #add keyword and location info
        if add_keyword is True:
            kw = self.train['keyword']
            loc = kw == kw
            kw.loc[loc] = 'KW:' + kw.loc[loc] + ' '
            kw = kw.fillna('')
            combined = kw + combined

        if add_loc is True:
            lc = self.train['location']
            loc = lc == lc
            lc.loc[loc] = 'LOC:' + lc.loc[loc] + ' '
            lc = lc.fillna('')
            combined = lc + combined

        self.X = combined.values
        self.y = self.train['target'].values



    def split_train_test(self,verbose = True):
        '''

        :return:
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size = 0.33,
                                                                                random_state = 42)
        if verbose is True:
            print('training data composition...')
            print(pd.Series(self.y_train).value_counts())
            print()
            print('test data composition...')
            print(pd.Series(self.y_test).value_counts())
            print()


    def gridsearch_model(self,parameters = {'vect__ngram_range': [(1, 1),
                                                                  (1, 2)],
                                     'tfidf__use_idf': (True,
                                                        False),
                                     'clf__alpha': (1e-2,
                                                    1e-3)},
                         verbose = True):
        '''now try SVM model'''
        #self.model.fit(self.X_train, self.y_train)
        #self.predicted = self.model.predict(self.X_test)
        self.model_gs = GridSearchCV(self.model, parameters, cv=5, n_jobs=-1)
        self.model_gs = self.model_gs.fit(self.X_train, self.y_train)
        if verbose is True:
            for param_name in sorted(parameters.keys()):
                print("%s: %r" % (param_name, self.model_gs.best_params_[param_name]))


    def print_performance_metrics(self,predicted,target,names=None):
        '''
        :return:
        '''
        print(metrics.classification_report(target, predicted,target_names=names))







if __name__ == '__main__':
    cl =nlp_model()
    cl.load_data(verbose = False)
    cl.assemble_X_y(add_loc = True,add_keyword = True)
    cl.split_train_test(verbose = False)
    cl.gridsearch_model()

    truths = cl.y_test
    predictions = cl.model_gs.predict(cl.X_test)
    print(metrics.classification_report(truths,predictions))

    #report some of the failed predictions
    idx_fail = predictions != cl.y_test
    X_test_fail = cl.X_test[idx_fail]
    pd.DataFrame({'text':X_test_fail,
                  'label':cl.y_test[idx_fail]}).to_csv('failures.csv')

    #plot confusion matrix
    plt.close()
    plot_confusion_matrix(cl.model_gs, cl.X_test, cl.y_test,
                          #display_labels=class_names,
                          cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.pdf')





