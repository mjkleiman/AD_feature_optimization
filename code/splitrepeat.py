import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, roc_auc_score, roc_curve, auc,  accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import label_binarize
from tensorflow import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


def splitrepeat_cv(X, y, model, splits, repeats, num_classes=2, class_labels=None, test_ratio=0.25, imbalanced=None, categorical_features=None, over_strategy='auto', under_strategy='auto', 
                   avg_strategy='macro', verbose=0, initial_split_seed=None, initial_split_ratio=0.25):
    '''
    Runs a j-split k-repeat random sub-sampling cross-validation for classification tasks

    model must be a sklearn model

    Parameters
    ----------
    X : pandas DataFrame
        Independent data
    y : pandas Series or numpy array
        Depndent data or labels
    model :  any scikit-learn classifier 
        Currently tested with randomforestclassifier, gradientboostedclassifier
    splits :  array
        Specify a list of seed values to be used. 
        TO DO: insert int to randomly assign seeds
    repeats :  array
        Specify a list of seed values to be used. 
        TO DO: insert int to randomly assign seeds
    num_classes : int
        Number of classes. If classes are not arranged in numerical format (ex: 0,1,2) then specify class_labels
    class_labels : list of strings or ints
        Set labels of classes if not numerical from 0
    test_ratio : float
        Used in sklearn.metrics.train_test_split to calculate the proportion of validation and test sets vs training data. 
        Test set is calculated first, followed by validation set, so if the same number is used for both the test set will be larger than the validation set.
    imbalanced : default=None
        'over' : utilize imblearn's SMOTE (or SMOTENC if categorical_features are defined) to oversample the train set
        'under' : utilize imblearn's EditedNearestNeighbours to undersample the test set
    categorical_features : list of categorical features in data, used in SMOTENC
    avg_strategy : see 'average' in sklearn's roc_auc_score (default = 'macro')
    verbose : 0, 1, or 2
        0 : disables all output
        1 : shows split/repeat number
        2 : adds confusion_matrix
    initial_split_seed : int
        If this value is specified, data will be initially split once. Use this to match previously used train/test splits (sklearn implementation) and to ensure that training data remains
        in the training set. Data on the testing side of the split may be shuffled into the training set, but never the reverse.
        If this value is not specified, all data will be shuffled. This is useful if a holdout test set will be used for final testing (note: do not test holdout sets using splitrepeat_cv)
    initial_split_ratio : float
        If initial_split_seed is specified, this ratio will be used to split initial train/test ratios. Small train splits are preferred to enable more data to be shuffled and to reduce overfitting.
        This value replaces "train_size" in sklearn's train_test_split.
        Note that the train data from this initial split will be added to all training sets generated 
    over_strategy : see "search_strategy" from imblearn.oversampling.SMOTE
    under_strategy : see "search_strategy" from imblearn.undersampling.EditedNearestNeighbours

    Returns
    -------
    Dataframe with sensitivity, specificity, PPV, NPV, accuracy, and F1 values for each class 
    '''
    df = pd.DataFrame()

    if class_labels == None:
        class_labels = list(range(num_classes)) # For multiclass ROC curve calculations (requires numerical input)

    if initial_split_seed != None:
        _X_train, X, _y_train, y = train_test_split(X, y.values.ravel(), train_size=initial_split_ratio, random_state=initial_split_seed, stratify=y)
        y = pd.Series(y)

    # Begin j-split k-repeat loop
    for j in splits:
        X_, X_test, y_, y_test = train_test_split(X, y.values.ravel(), test_size=test_ratio, random_state=j, stratify=y)
        if initial_split_seed != None:
            X_ = X_.append(_X_train)
            y_ = np.append(y_,_y_train)

        if imbalanced == 'under':
            enn = EditedNearestNeighbours(sampling_strategy=under_strategy, random_state=j)
            X_,y_ = enn.fit_resample(X_,y_)
            X_test,y_test = enn.fit_resample(X_test,y_test) # Add option to call test resampling
        if imbalanced == 'over':
            if categorical_features is None:
                sm = SMOTE(random_state=j, sampling_strategy=over_strategy)
            else:
                sm = SMOTENC(categorical_features = categorical_features, sampling_strategy=over_strategy, random_state=j)
            X_, y_ = sm.fit_resample(X_,y_)

        # Run models
        for k in repeats:
            np.random.seed(k)
            model.set_params(random_state=k)
            model.fit(X_,y_)
            y_pred = model.predict_proba(X_test)

            cmat = multilabel_confusion_matrix(np.array(y_test), np.argmax(y_pred, axis=1))
            if verbose >= 1:
                print('Split: ',j,', Repeat: ',k)
            if verbose >= 2:
                print(cmat)
            report = pd.DataFrame({'j':j,'k':k}, index=[0]) 
            for c in range(num_classes):                
                TP = cmat[c][1][1]
                FP = cmat[c][0][1]
                TN = cmat[c][0][0]
                FN = cmat[c][1][0]
                                
                report['Sensitivity'+str(c)] = (TP/(TP+FN))
                report['Specificity'+str(c)] = (TN/(FP+TN))
                report['PPV'+str(c)] = (TP/(TP+FP))
                report['NPV'+str(c)] = (TN/(TN+FN))
                report['Accuracy'+str(c)] = (TP+TN)/len(y_test)

            y_test1 = y_test.copy()
            y_test1[y_test1 > 1] = 1 #Set class 2 predictions to class 1, to emulate impairment detection
            y_pred = np.argmax(y_pred, axis=1)
            y_pred[y_pred > 1] = 1

            report['Sensitivity'] = recall_score(y_test1,y_pred, average=avg_strategy, labels=[1])
            report['Specificity'] = recall_score(y_test1,y_pred, average=avg_strategy, labels=[0])
            report['PPV'] = precision_score(y_test1,y_pred, average=avg_strategy, labels=[1])
            report['NPV'] = precision_score(y_test1,y_pred, average=avg_strategy, labels=[0])
            report['F1_Score'] = f1_score(y_test1,y_pred, average=avg_strategy)
            report['Accuracy'] = accuracy_score(y_test1, y_pred)

            report.set_index(['j','k'], inplace=True)

            df = df.append(report)

    return df

def splitrepeat_mcn(X, y, model_list, splits, repeats, num_classes, feature_list=None, mc_strategy='ovr', test_ratio=0.25, class_labels=None,
                    stacked_model=None, imbalanced=None, categorical_features=None, over_strategy='auto', under_strategy='auto', avg_strategy='macro',
                    initial_split_seed=None, initial_split_ratio=0.5, verbose=0):
    '''
    Runs a j-split k-repeat random sub-sampling cross-validation for classification tasks

    model must be a sklearn model

    Parameters
    ----------
    X : pandas DataFrame
        Independent data
    y : pandas Series or numpy array
        Depndent data or labels
    model :  any scikit-learn classifier 
        Currently tested with randomforestclassifier, gradientboostedclassifier
    splits :  array
        Specify a list of seed values to be used. 
        TO DO: insert int to randomly assign seeds
    repeats :  array
        Specify a list of seed values to be used. 
        TO DO: insert int to randomly assign seeds
    num_classes : int
        Number of classes. If classes are not arranged in numerical format (ex: 0,1,2) then specify class_labels
    class_labels : list of strings or ints
        Set labels of classes if not numerical from 0
    test_ratio : float
        Used in sklearn.metrics.train_test_split to calculate the proportion of validation and test sets vs training data. 
        Test set is calculated first, followed by validation set, so if the same number is used for both the test set will be larger than the validation set.
    imbalanced : default=None
        'over' : utilize imblearn's SMOTE (or SMOTENC if categorical_features are defined) to oversample the train set
        'under' : utilize imblearn's EditedNearestNeighbours to undersample the test set
    categorical_features : list of categorical features in data, used in SMOTENC
    avg_strategy : see 'average' in sklearn's roc_auc_score (default = 'macro')
    verbose : 0, 1, or 2
        0 : disables all output
        1 : shows split/repeat number
        2 : adds confusion_matrix
    initial_split_seed : int
        If this value is specified, data will be initially split once. Use this to match previously used train/test splits (sklearn implementation) and to ensure that training data remains
        in the training set. Data on the testing side of the split may be shuffled into the training set, but never the reverse.
        If this value is not specified, all data will be shuffled. This is useful if a holdout test set will be used for final testing (note: do not test holdout sets using splitrepeat_cv)
    initial_split_ratio : float
        If initial_split_seed is specified, this ratio will be used to split initial train/test ratios. Small train splits are preferred to enable more data to be shuffled and to reduce overfitting.
        This value replaces "train_size" in sklearn's train_test_split.
        Note that the train data from this initial split will be added to all training sets generated 
    over_strategy : see "search_strategy" from imblearn.oversampling.SMOTE
    under_strategy : see "search_strategy" from imblearn.undersampling.EditedNearestNeighbours

    Returns
    -------
    Dataframe with sensitivity, specificity, PPV, NPV, accuracy, and F1 values for each class
    '''
    df = pd.DataFrame()

    if class_labels == None:
        class_labels = list(range(num_classes)) # For multiclass ROC curve calculations (requires numerical input)

    if initial_split_seed != None:
        _X_train, X, _y_train, y = train_test_split(X, y.values.ravel(), train_size=initial_split_ratio, random_state=initial_split_seed, stratify=y)
        y = pd.Series(y)
    
    # Begin j-split k-repeat loop
    for j in splits:
        X_, X_test, y_, y_test = train_test_split(X, y.values.ravel(), test_size=test_ratio, random_state=j, stratify=y)
        if initial_split_seed != None:
            X_ = X_.append(_X_train)
            y_ = np.append(y_,_y_train)

        if imbalanced == 'under':
            enn = EditedNearestNeighbours(sampling_strategy=under_strategy, random_state=j)
            X_,y_ = enn.fit_resample(X_,y_)
            X_test,y_test = enn.fit_resample(X_test,y_test) # Add option to call test resampling
        if imbalanced == 'over':
            if categorical_features is None:
                sm = SMOTE(random_state=j, sampling_strategy=over_strategy)
            else:
                categorical_features = np.in1d(X_.columns.values, categorical_features)
                sm = SMOTENC(categorical_features = categorical_features, sampling_strategy=over_strategy, random_state=j)
            X_, y_ = sm.fit_resample(X_,y_)

        # Run models
        for k in repeats:
            np.random.seed(k)
            y_output = pd.DataFrame()
            y_ = pd.DataFrame(y_)
            for i in class_labels:
                model = model_list[i]
                model.set_params(random_state=k)
                X_i = X_[feature_list[i]] # Select feature list
                replace_y = {class_labels[i]:1}
                replace_y.update(zip([x for x in class_labels if x!=i],[0 for x in [x for x in class_labels if x!=i]]))
                y_i = y_.replace(replace_y) # Set selected class=1, others=0 (OneVsRest)
            
                model.fit(X_i,y_i.values.ravel())
                y_pred = model.predict_proba(X_test[feature_list[i]])
                y_output['Clf'+str(i)] = y_pred[:,1]

            # Use each classifier's target class (OneVsRest) output as selected output probability, then divide by total 
            # so that probability outputs sum to 1
            for i in range(len(y_output)):                         
                y_output.iloc[i,:] = y_output.iloc[i,:].divide(y_output.sum(axis=1)[i]).to_numpy() 
            
            if stacked_model != None:
                y_output = stacked_model.predict_proba(y_output)

            cmat = multilabel_confusion_matrix(y_test, np.argmax(y_output.to_numpy(), axis=1))
            if verbose >= 1:
                print('Split: ',j,', Repeat: ',k)
            if verbose >= 2:
                print(cmat)
            report = pd.DataFrame({'j':j,'k':k}, index=[0]) 

            for c in range(num_classes):                
                TP = cmat[c][1][1]
                FP = cmat[c][0][1]
                TN = cmat[c][0][0]
                FN = cmat[c][1][0]
                                
                report['Sensitivity'+str(c)] = (TP/(TP+FN))
                report['Specificity'+str(c)] = (TN/(FP+TN))
                report['PPV'+str(c)] = (TP/(TP+FP))
                report['NPV'+str(c)] = (TN/(TN+FN))
                report['Accuracy'+str(c)] = (TP+TN)/len(y_test)

            y_test1 = y_test.copy()
            y_test1[y_test1 > 1] = 1 #Set class 2 predictions to class 1, to enable comparison with two-class 
            y_pred = np.argmax(y_output.to_numpy(), axis=1)
            y_pred[y_pred > 1] = 1

            report['Sensitivity'] = recall_score(y_test1,y_pred, average=avg_strategy, labels=[1])
            report['Specificity'] = recall_score(y_test1,y_pred, average=avg_strategy, labels=[0])
            report['PPV'] = precision_score(y_test1,y_pred, average=avg_strategy, labels=[1])
            report['NPV'] = precision_score(y_test1,y_pred, average=avg_strategy, labels=[0])
            report['F1_Score'] = f1_score(y_test1,y_pred, average=avg_strategy)
            report['Accuracy'] = accuracy_score(y_test1, y_pred)

            report.set_index(['j','k'], inplace=True)

            df = df.append(report)

    return df