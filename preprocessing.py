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

def normality(dataframe: pd.DataFrame, alpha: float) -> bool:
    numerical_columns = dataframe.drop('Outcome', axis=1).columns

# Perform Shapiro-Wilk test for normality for each numerical column
    for column in numerical_columns:
        # Perform Shapiro-Wilk test
        statistic, p_value = shapiro(df[column])
    
        # # Print the results
        # print(f"Shapiro-Wilk test for column '{column}':")
        # print(f"Test Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    
        # Interpret the results
        
        if p_value > alpha:
            return True
        else:
            return False
        
def clus_detection(epsilon: int, min_samples: int, dataframe: pd.DataFrame, graph: bool)-> np.ndarray:
    
    outliers = []
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)

    numerical_columns = dataframe.drop('Outcome', axis=1).columns
    for column in numerical_columns:
    # Extract column data as a NumPy array
        column_data = dataframe[column].values.reshape(-1, 1)
    
    # Instantiate DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit DBSCAN to the column data
        dbscan.fit(column_data)
    
    # Extract labels and core sample indices
        labels = dbscan.labels_
    
    # Identify outliers
        outlier_indices = np.where(labels == -1)[0]
        for idx in outlier_indices:
            outliers.append([column, idx, column_data[idx][0]])
    
    outlier_array = np.array(outliers, dtype=object)



    if (graph):
        for column in numerical_columns:
            plt.figure(figsize=(8, 6))
    
            # All points
            plt.scatter(np.arange(len(column_data)), column_data, c=labels, cmap='viridis')
        
            # Highlight outliers
            plt.scatter(outlier_indices, column_data[outlier_indices], c='red', label='Outliers')
        
            plt.title(f'DBSCAN Outlier Detection on {column}')
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.colorbar(label='Cluster Label')
            plt.legend()
            plt.show()
    return outlier_array