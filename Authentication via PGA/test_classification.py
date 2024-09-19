from EEG_classification import Classification 
import numpy as np
import os
import pandas as pd
import optuna

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
    cross_val_score
)


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path).split('\\')
path = ""
for i in range(len(dir_path)):
    path = path + dir_path[i] + "\\"
file = path + "MNE_pipeline_with_muscle_features_familiar_start_trial.csv"
df = pd.read_csv(file).iloc[:,1:]

classifier = Classification(data=df)
X,Y = classifier.get_data()

X_smote,Y_smote = classifier.augmentation_SMOTE(X,Y)
X_adasyn,Y_adasyn = classifier.augmentation_ADASYN(X,Y)

params = {
    'C':[2,3,4,5,6,7,8,9,10,20,30,40,50,60],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],  
    'gamma':['scale','auto']
}

svm = classifier.svm(params=None,prob=True)

# params = {
#         'C':optuna.distributions.IntDistribution(2,60),
#         'kernel':optuna.distributions.CategoricalDistribution(['linear', 'poly', 'rbf', 'sigmoid']),
#         'gamma':optuna.distributions.CategoricalDistribution(['scale','auto'])
#     }   

params = {
    'C':[2,3,4,5,6,7,8,9,10,20,30,40,50,60],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],  
    'gamma':['scale','auto']
}

kfold = classifier.cross_validation_func(method='stratified',num_folds=5,shuffle=True)

# best_params = classifier.hyperparameter_tuning_optuna(X,Y,model=svm,sampler='TPE',params=params,fold=kfold,n_trials=200,metric='f1',)

best_params = classifier.hyperparameter_tuning_gridsearch(X,Y,model=svm,params=params,fold=kfold,train_score=True)

model = classifier.svm(params=best_params,prob=True)

print(classifier.cross_validation(X,Y,model=model,fold=kfold,metrics=['accuracy','precision','recall','f1_score','auc_pr_curve','eer']))