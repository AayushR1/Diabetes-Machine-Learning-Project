import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer

from sklearn.cluster import DBSCAN

from scipy.stats import shapiro

from sklearn.metrics import confusion_matrix, make_scorer

from sklearn.model_selection import cross_val_score, StratifiedKFold


def loss_function (y_true: pd.DataFrame, y_pred: pd.DataFrame) -> int:
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    fn = cm[1, 0]
    print("False Positive", fp)
    print("false Neg", fn)
    return fp + 100 * fn

custom_scorer = make_scorer(loss_function, greater_is_better=True)

def lr_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test:pd.Series, labels: list, score: bool)-> int:
    """
    Train and evaluate Logistic Regression, K-Nearest Neighbors, and Random Forest models, and plot confusion matrices.
    
    Parameters:
    X_train, X_test (DataFrame): Training and test predictors
    y_train, y_test (Series): Training and test target variable
    labels (list): List of class labels
    
    Returns:
    None
    """
    # Initialize models
    classifier_lr = LogisticRegression(max_iter=1000, random_state=42)
 

    # Fit models
    classifier_lr.fit(X_train, y_train)


    # Predict
    y_pred_lr = classifier_lr.predict(X_test)

    # Calculate accuracy scores
    score_lr = classifier_lr.score(X_test, y_test)

    
    # Compute confusion matrices
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)


    # Plot confusion matrices
    plot_confusion_matrix(cm_lr, labels, title='Confusion Matrix: Logistic Regression')
    
    TN, FP, FN, TP = cm_lr.ravel()

    cm_score = FP + (FN*100)
    if score:
        return cm_score
    





def kn_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, labels: list, score: bool)-> int:
    """
    Train and evaluate K-Nearest Neighbors model, and plot confusion matrix.
    
    Parameters:
    X_train, X_test (DataFrame): Training and test predictors
    y_train, y_test (Series): Training and test target variable
    labels (list): List of class labels
    score (bool): Whether to return the accuracy score
    
    Returns:
    float: Accuracy score if score is True, otherwise None
    """
    # Initialize model
    classifier_kn = KNeighborsClassifier()

    # Fit model
    classifier_kn.fit(X_train, y_train)

    # Predict
    y_pred_kn = classifier_kn.predict(X_test)

    # Calculate accuracy score
    score_kn = classifier_kn.score(X_test, y_test)

    # Compute confusion matrix
    cm_kn = confusion_matrix(y_test, y_pred_kn, labels=labels)

    # Plot confusion matrix
    plot_confusion_matrix(cm_kn, labels, title='Confusion Matrix: K-Nearest Neighbors')
    
    TN, FP, FN, TP = cm_kn.ravel()

    cm_score = FP + (FN*100)
    if score:
        return cm_score
    


def knn_fold(min_range: int, max_range: int, strat_num: int, X: np.ndarray, y: pd.Series):

    k_values = range(min_range,  max_range)

    results = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv = StratifiedKFold(n_splits=strat_num, shuffle=True, random_state=10)
        scores = cross_val_score(knn, X, y, cv=cv, scoring= custom_scorer)
        results.append((k, scores.mean()))
    
    results_dataframe = pd.DataFrame(results, columns=['k', 'loss'])

    best_k = results_dataframe.loc[results_dataframe['loss'].idxmin()]

    return best_k, best_k['loss']
    

def rf_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, labels: list, score: bool)-> int:

    """
    Train and evaluate Random Forest model, and plot confusion matrix with counts.
    
    Parameters:
    X_train, X_test (DataFrame): Training and test predictors
    y_train, y_test (Series): Training and test target variable
    labels (list): List of class labels
    
    Returns:
    int: Custom score based on FP and FN counts
    """
    # Initialize model
    classifier_rf = RandomForestClassifier(random_state=42)

    # Fit model
    classifier_rf.fit(X_train, y_train)

    # Predict
    y_pred_rf = classifier_rf.predict(X_test)

    # Compute confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels)

    score_rf = classifier_rf.score(X_test, y_test)

    TN, FP, FN, TP = cm_rf.ravel()

    cm_score = FP + (FN*100)
    if score:
        return cm_score
    

def rf_fold(min_range_ne: int, max_range_ne: int, min_range_md: int, max_range_md: int, strat_num: int, X: pd.DataFrame, y: pd.Series):
    n_estimators_values = range(min_range_ne, max_range_ne, 10)
    max_depth_values = range(min_range_md, max_range_md)

    results = []

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            print(rf);
            cv = StratifiedKFold(n_splits=strat_num, shuffle=True, random_state=42)
            scores = cross_val_score(rf, X, y, cv=cv, scoring=custom_scorer)
            print(scores.mean())
            results.append((n_estimators, max_depth, scores.mean()))

    results_dataframe = pd.DataFrame(results, columns=['n_estimators', 'max_depth', 'loss'])

    best_ne_md = results_dataframe.loc[results_dataframe['loss'].idxmin()]

    return best_ne_md['n_estimators'], best_ne_md['max_depth'], best_ne_md['loss']


def plot_confusion_matrix(cm, labels: list, title='Confusion Matrix', cmap= "Greens"):
    """
    Plots a confusion matrix using seaborn heatmap.

    Parameters:
    cm (array): Confusion matrix
    labels (list): List of label names to index the matrix.
    title (str): Title for the plot
    cmap (str): Colormap for the heatmap

    Returns:
    None
    """
    plt.figure(figsize=(10, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()