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

columns_to_replace_zeros = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']


def imputing(n_neighbors: int, dataframe: pd.DataFrame)->np.ndarray:
    for column_to_replace_zeros in columns_to_replace_zeros:
        zero_mask = dataframe[column_to_replace_zeros] == 0
        dataframe.loc[zero_mask, column_to_replace_zeros] = float('nan')
    X = dataframe.values
    imputer = KNNImputer(n_neighbors = n_neighbors)
    X_imputed = imputer.fit_transform(X)

    return X_imputed

def clus_detection(epsilon: int, min_samples: int, dataframe: pd.DataFrame)