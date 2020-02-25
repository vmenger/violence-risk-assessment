# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:03:04 2020

@author: Moste007
"""

import pandas as pd
import sklearn.model_selection as modsel
import numpy as np
import joblib
from sklearn import svm

log_string = ""

def log(message):
    global log_string 
    log_string += message + "\n"

def setup_predictions():
    # Initialize empty dataframe that will hold predictions and outcomes, to compute statistics
    predictions = pd.DataFrame(columns=['probability', 'true_label', 'fold_number'])
    return predictions

def fit(predictions, no_inner_folds, no_outer_folds, X, y, groups, param_grid):
    # Define fold_counter
    fold_counter = 0
    
    # Iterate over outer folds for internal cross validation
    fold_generator = modsel.GroupKFold(n_splits=no_outer_folds).split(X, y, groups)
    for train_indices, test_indices in fold_generator:
    
        # Increase fold_counter
        fold_counter += 1
        
        log("Outer fold {}".format(fold_counter))
        
        # Subset data based on folds indices
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        groups_train = groups[train_indices]
        test_admission_ids = notes['admission_id'][test_indices]
    
        # Determine number of models
        no_models = np.prod([len(l) for _, l in param_grid.items()])
        log("\t Fitting {} models with {}-dimensional table".format(no_models * no_inner_folds, 
                                                                    X_train.shape))
        
        # Get estimator object
        grid_estimator = get_estimator(no_inner_folds, param_grid)
        
        # Fit model
        grid_estimator.fit(X_train, y_train, groups_train)
        
        # Log results
        log("\t Best parameters: {}".format(grid_estimator.best_params_))
        log("\t CV result: {:.3f}+/-{:.3f}".format(grid_estimator.best_score_,
                                                   grid_estimator.cv_results_['std_test_score'][grid_estimator.best_index_ ]))
        log("\t Full report: {}".format(grid_estimator.cv_results_))
        log("===============================")
        
        # Extract best estimator after training
        best_model = grid_estimator.best_estimator_
        
        # Store predictions and true labels in a dataframe
        df = pd.DataFrame({'probability' : best_model.predict_proba(X_test)[:, 1], 
                           'true_label' : y_test,
                           'admission_id' : test_admission_ids,
                          })
        
        # Add fold counter
        df['fold_number'] = fold_counter
        
        # Append to the dataframe that will hold all predictions
        predictions = pd.concat([predictions, df], axis=0, sort=True)
        
    # Reset index, recode label    
    predictions = predictions.reset_index().drop(['index'], axis=1)
    predictions['true_label'] = predictions['true_label'].map({0 : False, 1 : True})
    
    # Write predictions for evaluation
    predictions.to_csv("logs/predictions.csv", 
                       sep=";", 
                       index=False)
    
    # Write log
    with open ("logs/log.txt", "w+") as logfile:
        logfile.write(log_string)
        
def save_model(X, y, groups, no_inner_folds, param_grid):
    # Train model on entire dataset for external evaluation
    grid_estimator = get_estimator(no_inner_folds, param_grid)
    grid_estimator.fit(X, y, groups)
    
    # Write model to disk
    joblib.dump(grid_estimator.best_estimator_, 'models/svm_model')

def get_estimator(no_inner_folds, param_grid):
    # Define support vector machine instance
    svm_model = svm.SVC(kernel='rbf',
                        class_weight='balanced',
                        probability=True)

    # Define inner split object
    inner_split_object = modsel.GroupKFold(n_splits=no_inner_folds)

    # Define grid search object
    grid_estimator = modsel.GridSearchCV(estimator=svm_model,
                                  param_grid = param_grid,
                                  scoring='roc_auc',
                                  cv=inner_split_object,
                                  return_train_score=False,
                                  refit=True,
                                  verbose=0
                                 )
    
    return(grid_estimator)    
    
def train_classifier(notes):
    # Shuffle dataset
    notes = notes.sample(frac=1)
    
    # Note vectors are used as input (X), apply zscore normalization
    X = notes[[str(a) for a in range(300)]].values
    
    # Target label (y) is defined in the outcome variable
    y = notes['outcome'].values
    
    # groups are defined by patient identifiers
    groups = notes['patient_id'].values
    
    # Define number of inner and outer folds
    no_inner_folds = 5
    no_outer_folds = 5
    
    # Define a parameter grid for optimizing hyperparameters
    param_grid = {'kernel' : ['rbf'],
                  'C'      : [1e-1, 1e0, 1e1],
                  'gamma'  : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
                 }
    
    log("Using search grid {}".format(param_grid))
    log("Number of inner folds = {}".format(no_inner_folds))
    log("Number of outer folds = {}".format(no_outer_folds))
    log("===============================")
    
    predictions = setup_predictions()
    
    fit(predictions, no_inner_folds, no_outer_folds, X, y, groups, param_grid)
    
    save_model(X, y, groups, no_inner_folds, param_grid)
    
if __name__ == "__main__":
    notes = pd.read_csv("data/processed/notes.csv", sep=";")
    train_classifier(notes)
    