from sklearn.model_selection import KFold
import pickle
import numpy as np
import os
from sklearn import metrics
import matplotlib.pylab as plt

'''
Build a K-fold cross validation tool compare models
'''

def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(n_splits=3, shuffle=True)
    y_pred = np.zeros(len(y))*0.0


    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf_class.fit(X_train, y_train,**kwargs)
        #first try to return class prbabilities
        try:
            yp = clf_class.predict_proba(X_test)
        except:
            print('predict_proba function not present, returning raw predictions')
            yp = clf_class.predict(X_test)
        if len(np.shape(yp)) > 1:
            yp = yp[:,-1]
        y_pred[test_index] = yp
    return y_pred

def perform_benchmarking():
    model_paths = ['./models/naive_bayes.pickle',
                   './models/lstm.pickle']
    model_meta = {'model': [],
                  'name':[],
                  'X_train': [],
                  'y_train': [],
                  'X_test': [],
                  'y_test': [],
                  'y_pred_concat': [],
                  'y_test_concat':[],
                  'kwargs': []}
    for m in model_paths:
        pickle_in = pickle.load(open(m, "rb"))
        model_meta['model'].append(pickle_in['model'])
        model_meta['X_train'].append(pickle_in['X_train'])
        model_meta['y_train'].append(pickle_in['y_train'])
        model_meta['X_test'].append(pickle_in['X_test'])
        model_meta['y_test'].append(pickle_in['y_test'])
        model_meta['kwargs'].append(pickle_in['kwargs'])
        model_meta['name'].append(pickle_in['model_name'])
        if type(pickle_in['X_train']) == np.ndarray:
            X = np.vstack([np.array(pickle_in['X_train']), np.array(pickle_in['X_test'])])
        else:
            X = np.array(pickle_in['X_train'] + pickle_in['X_test'])
        if type(pickle_in['y_train']) == np.ndarray:
            y = np.vstack([np.array(pickle_in['y_train']), np.array(pickle_in['y_test'])])
        else:
            y = np.array(pickle_in['y_train'] + pickle_in['y_test'])
        y_pred = run_cv(X, y, pickle_in['model'], **pickle_in['kwargs'])
        model_meta['y_pred_concat'].append(y_pred)
        model_meta['y_test_concat'].append(y)
    # save benchmarking results
    picklefile = './models/benchmarking_model_predictions.pickle'
    os.system('rm ' + picklefile)
    pickle_out = open(picklefile, "wb")
    pickle.dump(model_meta, pickle_out)
    pickle_out.close()

    return model_meta


if __name__ == '__main__':
    '''
    Perform K-fold CV on previously fitted models
    '''
    #model_meta = perform_benchmarking()

    '''
    Load benchmarking results
    '''
    model_meta = pickle.load(open('./models/benchmarking_model_predictions.pickle', "rb"))


    '''
    construct the roc curve for both models
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(len(model_meta['X_train'])):
        y_test = model_meta['y_test_concat'][i]
        y_pred = model_meta['y_pred_concat'][i]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print('ROC curve AUC = ' + str(auc))
        idx = np.argsort(fpr)
        fpr = fpr[idx]
        tpr = tpr[idx]
        ax1.plot(fpr,tpr,label=str(model_meta['name'][i])+'\nAUC:'+str(np.round(auc,2)))
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    plt.tight_layout()
    plt.legend()
    plt.savefig('ROC_curve_model_comparison.pdf')











