from sklearn.cross_validation import KFold
import pickle

'''
Build a K-fold cross validation tool compare models
'''

def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=3, shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred



if __name__ == '__main__':
    '''
    Load previously fitted models
    '''
    model_paths = ['./models/naive_bayes.pickle',
              './models/lstm.pickle']
    model_meta = {'model': [],
             'X_train': [],
             'y_train': [],
             'X_test': [],
             'y_test': []}
    for m in model_paths:
        pickle_in = pickle.load(open(m, "rb"))
        model_meta['model'].append(pickle_in['model'])
        model_meta['X_train'].append(pickle_in['X_train'])
        model_meta['y_train'].append(pickle_in['y_train'])
        model_meta['X_test'].append(pickle_in['X_test'])
        model_meta['y_test'].append(pickle_in['y_test'])

        y_pred = run_cv(X, y, clf_class, **kwargs)





