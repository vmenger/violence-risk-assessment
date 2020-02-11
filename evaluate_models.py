# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:27:20 2020

@author: Moste007
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import delong_auc as da
import sklearn.metrics as sklm
import scipy as scp
import gensim.models as gsm
import joblib
import process_text as pt

# Determine a binary cutoff for a group of predictions
def binary_cutoff(group):
    
    # Ratio of admissions with a positive outcome in this group
    ratio = 1 - (sum(group['true_label']) / len(group['true_label']))
    
    # Determine threshold based on this ratio
    threshold = sorted(group['probability'])[int(ratio * len(group))]
    group['binary_prediction'] = (group['probability'] > threshold)
    
    # Return
    return(group)

def evaluate_models(out_fig_file_name):
    # Read files
    predictions = pd.read_csv("logs/predictions.csv", sep=";")
    
    #Evaluate cross-validation prediction (internal)
    # Create empty figure
    plt.figure(figsize=(7, 5), dpi=450)
    
    # Track values over multiple folds
    aucs = []
    auc_vars = []
    
    # Determine number of folds
    no_folds = predictions['fold_number'].nunique()
    
    # For each fold
    for i in np.arange(no_folds)+1:
        
        # Select subset of dataframe corresponding to fold
        predictions_fold = predictions[predictions['fold_number'] == i]
        
        # Compute area under curve and variance based on DeLong method
        auc, var = da.delong_roc_variance(predictions_fold['true_label'], 
                                       predictions_fold['probability'])
    
        # Track total
        aucs.append(auc)
        auc_vars.append(var)
        
        # Compute FPR and TPR rates for plotting
        fpr, tpr, thresholds = sklm.roc_curve(predictions_fold['true_label'], 
                                         predictions_fold['probability'])        
        
        # Add to plot
        plt.plot(fpr, tpr, label="Fold {} (AUC={:.3f})".format(i, auc))
        
    # Sampling distribution of the mean 
    auc_mean = np.mean(aucs)
    auc_var = np.mean(auc_vars)
    auc_ste = np.sqrt(auc_var) / np.sqrt(no_folds)
        
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title('AUROC Site x = {:.3f}'.format(auc_mean))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(out_fig_file_name)
        
    # Compute area under curve and variance based on DeLong method
    auc_delong, var_delong = da.delong_roc_variance(predictions['true_label'], 
                                   predictions['probability'])
        
    print("Auc = {:.5f}".format(auc_mean))
    print("Var = {:.5f}".format(auc_var))    
    print("Ste = {:.5f}".format(auc_ste))    
    print("95% CI = {}".format(scp.stats.norm.ppf([0.025, 0.975],loc=auc_mean, scale=auc_ste)))
    
    # A binary cutoff is determined for each fold
    predictions = predictions.groupby("fold_number").apply(binary_cutoff)
    
    # Show 2x2 contingency table
    pd.crosstab(predictions['true_label'], predictions['binary_prediction'])
    
def evaluate_external_models():
    # Read external paragraph2vec and svm models
    p2v_model_external = gsm.Doc2Vec.load("models_external/paragraph2vec_model")
    svm_model_external = joblib.load("models_external/svm_model")

    # Read processed notes
    notes = pd.read_csv("data/processed/notes.csv", sep=";")
    
    # Obtain vectors of notes using external paragraph2vec model
    note_vectors_external = pt.text_to_vectors(notes, 
                                               'words_stemmed', 
                                               p2v_model_external, 
                                               no_reps=10)

    # Predict probabilities using external classification model
    probability_external = svm_model_external.predict_proba(note_vectors_external)[:, 1]

    # Create dataframe with predictions
    predictions_external = pd.DataFrame({'probability' : probability_external, 
                                     'true_label'  : notes['outcome'].map({0 : False, 1 : True})})

    # All predictions in same 'fold'
    predictions_external['fold_number'] = 1

    # Determine binary cutoff
    predictions_external = predictions_external.groupby("fold_number").apply(binary_cutoff)

    # Compute area under curve and covariance based on DeLong method
    auc_external, auc_var_external = da.delong_roc_variance(predictions_external['true_label'], 
                                                 predictions_external['probability'])

    print("External auc = {:.3f}".format(auc_external))
    print("External ste = {:.3f}".format(np.sqrt(auc_var_external)))
    print("External 95% CI = {}".format(scp.stats.norm.ppf([0.025, 0.975],loc=auc_external, scale=np.sqrt(auc_var_external))))

    predictions_external.to_csv("logs/predictions_external.csv", sep=";", index=False)
    
if __name__ == "__main__":
    out_fig_file_name = "auc.png"
    evaluate_models(out_fig_file_name)
    evaluate_external_models()
    