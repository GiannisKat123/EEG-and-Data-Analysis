import os
import pandas as pd
import optuna
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,IsolationForest
import optuna_integration
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
    cross_val_score
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    make_scorer,
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
    cross_val_score
)

from collections import defaultdict
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class Classification:
    def __init__(self,data:pd.Series):
        self.data = data if type(data) == np.ndarray else data.to_numpy()
        self.X = self.data[:,0:-1]
        self.Y = self.data[:,-1]
        self.unique_classes = np.unique(self.Y)
        self.cols = list(data.columns)

    def get_data(self):
        return self.X,self.Y

    def augmentation_SMOTE(self,X,Y):
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, Y)
        return X_resampled, y_resampled
            
    def augmentation_ADASYN(self,X,Y):
        ada = ADASYN()
        X_resampled, y_resampled = ada.fit_resample(X, Y)
        return X_resampled, y_resampled
    
    def svm(self,params:dict|None,prob:bool):
        if params:
            model = SVC(**params,probability=prob)
        else:
            model = SVC(probability=prob)
        return model

    def decision_tree(self,params:dict|None):
        model = DecisionTreeClassifier(**params)
        return model
    
    def logistic_regression(self,params:dict|None):
        model = LogisticRegression(**params)
        return model

    def knn(self,params:dict|None):
        model = KNeighborsClassifier(**params)
        return model
    
    def lda(self,params:dict|None):
        model = LinearDiscriminantAnalysis(**params)
        return model
    
    def random_forest(self,params:dict|None):
        model = RandomForestClassifier(**params)
        return model
    
    def xgboost(self,params:dict|None):
        model = XGBClassifier(**params)
        return model
    
    def cross_validation_func(self,method:str,num_folds:int,shuffle:bool):
        if method == 'stratified':
            folds = StratifiedKFold(n_splits=num_folds,shuffle=shuffle,random_state=1)
            return folds
        else: return None

    def feature_reduction(self,X:np.ndarray,Y:np.array,model:str,num_of_features:int):
        if model == 'RandomForest':
            model = RandomForestClassifier(n_estimators=1000,random_state=1)
        model.fit(X,Y)
        importances = model.feature_importances_
        
        sorted_indices = np.argsort(importances)[::-1][:num_of_features]
            
        importance_list = []
        for id in sorted_indices:
            importance_list.append([self.cols[id],importances[id]])
        
        data = []
        
        for i in range(X.shape[0]):
            data_ = []
            for j in sorted_indices:
                data_.append(X[i][j])
            data.append(data_)
        
        X_new = np.array(data)        
        return X_new,Y,importance_list

    def hyperparameter_tuning_gridsearch(self,X:np.ndarray,Y:np.array,model,params:dict,fold,train_score:bool=True):
        hp_model = GridSearchCV(model,params,cv=fold,return_train_score=train_score,verbose=10,n_jobs=-1)
        hp_model.fit(X,Y)
        return hp_model.best_params_

    def hyperparameter_tuning_optuna(self,X:np.ndarray,Y:np.array,model,sampler:str,fold,params:dict,n_trials:int,metric:list[str]|str):
        if (not params or not model or not sampler): return 
        if sampler == 'TPE':
            sampler = optuna.samplers.TPESampler(seed=10)
        
        optuna_search = optuna_integration.OptunaSearchCV(
            model,
            params,
            cv=fold,
            n_trials = n_trials,
            scoring = metric,
            n_jobs=-1,
            random_state=10
        )

        optuna_search.fit(X,Y)
        return optuna_search.best_params_
    
    def train_test_split(self,X:np.ndarray,Y:np.array,train_test_split:tuple[float,float]|None,train_test_val_split:float|None):
        if train_test_val_split:
            X_train, X_rem, y_train, y_rem = train_test_split(X, Y, test_size=train_test_val_split(0), random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=train_test_val_split(1), random_state=42)
            return X_train,X_test,X_val,y_train,y_test,y_val
        elif train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_split, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            return None
        
    def train_model(self,model,Xtrain:np.ndarray,Ytrain:np.array):
        model.fit(Xtrain,Ytrain)
        return model
    
    def predict(self,model,Xtest:np.ndarray,Ytest:np.array):
        pred = model.predict(Xtest)
        return pred
    
    def cross_validation(self,X:np.ndarray,Y:np.array,model,fold,metrics:list[str]):
        mean_results = defaultdict(list)
        final_results = dict()
        for idx in fold.split(X,Y):
            train_idx,test_idx = idx[0],idx[1]
            Xtrain = X[train_idx]
            Ytrain = Y[train_idx]

            Xtest = X[test_idx]
            Ytest = Y[test_idx]

            model.fit(Xtrain,Ytrain)
            preds = model.predict(Xtest)

            results = self.calc_metrics(metrics,Ytest,preds)
            for key in results.keys():
                mean_results[key].append(results[key])
        for key in mean_results.keys():
            final_results[key] = np.mean(mean_results[key])
        return final_results

    def test_model(self,model,Xtest:np.ndarray,Ytest:np.array,metrics:list[str]):
        Ypred = model.predict(Xtest)
        return self.calc_metrics(metrics,Ytest,Ypred)
    
    def plot_pca(self,X:np.ndarray,Y:np.array,n_components:int,labels:list[str]):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
    
        ##### Plot the reduced data

        plt.figure(figsize=(10, 8))

        for i in range(len(np.unique(Y))):
            plt.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1])

        plt.xlabel('Principal Component 1',fontsize=30)
        plt.ylabel('Principal Component 2',fontsize=30)
        plt.legend()
        plt.show()

    def plot_pr_curve(self,list_of_recalls:list[float],list_of_precisions:list[float],list_of_aucs:list[float],label:str,title:str):
        plt.figure(figsize=(8, 6))
        for i in range(len(list_of_aucs)):
            plt.plot(list_of_recalls[i], list_of_precisions[i], marker='.', label=f'{label} (AUC = {list_of_aucs[i]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calc_metrics(self,metrics:list[str],Ytest:np.array,Ypred:np.array):
        # average='macro'
        results = dict()
        if 'accuracy' in metrics: results['accuracy'] = accuracy_score(Ytest,Ypred)
        if 'precision' in metrics:  results['precision'] = precision_score(Ytest,Ypred)
        if 'recall' in metrics: results['recall'] = recall_score(Ytest,Ypred)
        if 'f1_score' in metrics:  results['f1_score'] = f1_score(Ytest,Ypred)
        if 'auc_pr_curve' in metrics:
            precision,recall,_ = precision_recall_curve(Ytest, Ypred)
            results['auc_pr_curve'] = auc(recall,precision)
            results['precision_pr_curve'] = precision
            results['recall_pr_curve'] = recall
        if 'eer' in metrics:
            fpr, tpr, thresholds = roc_curve(Ytest, Ypred)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            far = fpr
            frr = 1 - tpr
            thresh = interp1d(fpr, thresholds)(eer)
            results['eer'] = eer
            results['far'] = far
            results['frr'] = frr
            results['threshold'] = thresh
        return results










