import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

class nlp_model:

    def __init__(self):
        self.train_file = './data/train.csv'
        self.test_file = './data/test.csv'
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, random_state=42,
                                  max_iter=5, tol=None)),
        ])


    def load_data(self):
        self.train = pd.read_csv(self.train_file)
        self.test  = pd.read_csv(self.test_file)

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


    def gridsearch_model(self,parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                                     'tfidf__use_idf': (True, False),
                                     'clf__alpha': (1e-2, 1e-3)}):
        '''now try SVM model'''
        #self.model.fit(self.X_train, self.y_train)
        #self.predicted = self.model.predict(self.X_test)
        self.model_gs = GridSearchCV(self.model, parameters, cv=5, n_jobs=-1)
        self.model_gs = self.model_gs.fit(self.X_train, self.y_train)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, self.model_gs.best_params_[param_name]))


    def print_performance_metrics(self,predicted,target,names=None):
        '''

        :return:
        '''
        print(metrics.classification_report(target, predicted,target_names=names))


if __name__ == '__main__':
    cl =nlp_model()
    cl.load_data()









