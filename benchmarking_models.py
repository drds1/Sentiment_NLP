from sklearn.model_selection import KFold
import pickle
import numpy as np
import os
'''
Build a K-fold cross validation tool compare models
'''

def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(n_splits=3, shuffle=True)
    y_pred = y.copy()


    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf_class.fit(X_train, y_train,**kwargs)
        yp = clf_class.predict(X_test)
        if len(np.shape(yp)) > 1:
            yp = yp[:,0]
        y_pred[test_index] = yp
    return y_pred

def perform_benchmarking():
    model_paths = ['./models/naive_bayes.pickle',
                   './models/lstm.pickle']
    model_meta = {'model': [],
                  'X_train': [],
                  'y_train': [],
                  'X_test': [],
                  'y_test': [],
                  'y_pred': [],
                  'kwargs': []}
    for m in model_paths:
        pickle_in = pickle.load(open(m, "rb"))
        model_meta['model'].append(pickle_in['model'])
        model_meta['X_train'].append(pickle_in['X_train'])
        model_meta['y_train'].append(pickle_in['y_train'])
        model_meta['X_test'].append(pickle_in['X_test'])
        model_meta['y_test'].append(pickle_in['y_test'])
        model_meta['kwargs'].append(pickle_in['kwargs'])
        if type(pickle_in['X_train']) == np.ndarray:
            X = np.vstack([np.array(pickle_in['X_train']), np.array(pickle_in['X_test'])])
        else:
            X = np.array(pickle_in['X_train'] + pickle_in['X_test'])
        if type(pickle_in['y_train']) == np.ndarray:
            y = np.vstack([np.array(pickle_in['y_train']), np.array(pickle_in['y_test'])])
        else:
            y = np.array(pickle_in['y_train'] + pickle_in['y_test'])
        y_pred = run_cv(X, y, pickle_in['model'], **pickle_in['kwargs'])
        model_meta['y_pred'].append(y_pred)

    return model_meta

if __name__ == '__main__':
    '''
    Perform K-fold CV on previously fitted models
    '''
    model_meta = perform_benchmarking()
    # save benchmarking results
    picklefile = 'benchmarking_model_predictions.pickle'
    os.system('rm ' + picklefile)
    pickle_out = open(picklefile, "wb")
    pickle.dump(model_meta, pickle_out)
    pickle_out.close()









