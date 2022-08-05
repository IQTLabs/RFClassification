## all models

from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import pickle
import time
from tqdm import tqdm
import numpy as np
from latency_helpers import *

from sklearn.metrics import ConfusionMatrixDisplay


class PsdSVM():
    def __init__(self, t_seg, nfft, C=1, gamma='scale'):
        self.t_seg = t_seg
        self.nfft = nfft
        self.C = C
        self.gamma = gamma
        self.svc = svm.SVC(kernel='rbf', C=self.C, gamma = self.gamma)
        
    def run_cv(self, X, y, k_fold=5):
        self.cv_acc_ls = []
        self.cv_f1_ls = []
        self.cv_runt_ls = []
        cv = KFold(n_splits=k_fold, random_state=None, shuffle=True)
        for i, (train_ix, test_ix) in tqdm(enumerate(cv.split(X))):
            svc = svm.SVC(kernel='rbf', C=self.C, gamma = self.gamma)
            svc.fit(X[train_ix], y[train_ix])
            
            t_start = time.time()
            y_pred = svc.predict(X[test_ix])
            t_dur = time.time()-t_start
            t_batch_avg = t_dur/len(y_pred)
            
            acc = accuracy_score(y[test_ix], y_pred)
            f1 = f1_score(y[test_ix], y_pred, average='weighted')
            
            print('Fold '+str(i+1)+': Accuracy: {:.3},\t F1: {:.3}, \t Runtime: {:.3}'.format(acc,f1, t_batch_avg))
            
#             # predict on the test data (by each sample)
#             y_pred, runtimes = atomic_benchmark_estimator(svc, X_use[test_ix], output_type='<U3', verbose=False)
            # Add to Lists
            self.cv_runt_ls.append(t_batch_avg)
            self.cv_acc_ls.append(acc)
            self.cv_f1_ls.append(f1)

        self.print_result(str(k_fold)+' Fold CV', self.cv_acc_ls, self.cv_f1_ls, self.cv_runt_ls)
        return self.cv_acc_ls, self.cv_f1_ls, self.cv_runt_ls
        
    def run_gridsearch(self, X, y, params, k_fold=5):
        self.grid_best_params_ls = []
        self.grid_acc_ls = []
        self.grid_f1_ls = []
        self.grid_runt_ls = []
        
        kf = KFold(n_splits=k_fold, random_state=None, shuffle=True)

        for i, (train_ix, test_ix) in enumerate(kf.split(X)):

            # find the optimal hypber parameters
            svc = svm.SVC(kernel='rbf', gamma = self.gamma)
            clf = GridSearchCV(svc, params, n_jobs=1)
            clf.fit(X[train_ix], y[train_ix])

            print('Fold '+str(i+1)+' Best Parameters: '+ str(clf.best_params_))
            self.grid_best_params_ls.append(clf.best_params_)

            # predict on the test data
            t_start = time.time()
#             y_pred = clf.predict(X[test_ix])
            
            y_pred, runtimes = atomic_benchmark_estimator(clf, X[test_ix], y.dtype, verbose=False) # predict & measure time
        
#             print(y_pred)
#             
            acc = accuracy_score(y[test_ix], y_pred)
#             print(y_pred)
#             print(y[test_ix])
            f1 = f1_score(y[test_ix], y_pred, average='weighted')
            print('Fold '+str(i+1)+': Accuracy: {:.3},\t F1: {:.3}, \t Runtime: {:.3}'.format(acc,f1, np.mean(runtimes)))
            
            # Append to List
            self.grid_acc_ls.append(acc)
            self.grid_f1_ls.append(f1)
            self.grid_runt_ls.append(np.mean(runtimes))
        
        self.print_result(str(k_fold)+' Fold GridSearch CV', self.grid_acc_ls, self.grid_f1_ls, self.grid_runt_ls)
        return self.grid_acc_ls, self.grid_f1_ls, self.grid_runt_ls, self.grid_best_params_ls
        
    
    def train(self, X_train, y_train):
        self.svc.fit(X_train, y_train)
        
    def predict(self, X):
        return self.svc.predict(X)
        
        
    def save(self, model_path, model_name):
        print('Model saved as:', model_path+model_name)
        pickle.dump(self.svc, open(model_path+model_name, 'wb'))
        
    def print_result(self, test_name, acc_ls, f1_ls, runt_ls):
        out_msg = 'PSD+SVM '+ test_name+ ' acc: {:.3}, F1: {:.3}, Run-time: {:.3}ms'.format(np.mean(acc_ls), np.mean(f1_ls), np.mean(runt_ls)*1e3)
        print(out_msg)
        
        
        
##### Visualization of Models #####
def show_confusion_matrix(dataset, labels, predictions, normalize='true', to_save=False, plot_folder='na', plot_name='na'):
    disp = ConfusionMatrixDisplay.from_predictions(labels, predictions, normalize=normalize)
#     display_labels=list(dataset.class_to_idx.keys())
    if to_save:
        disp.figure_.savefig(f"{plot_folder}/confusion_matrix_{plot_name}.png")
    
        
        