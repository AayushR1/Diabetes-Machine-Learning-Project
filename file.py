
#%%
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
from importlib import reload

from modeling import lr_models, rf_model, kn_model


#%%
df = pd.read_csv('diabetes.csv')

print(df.columns)


x = df['BloodPressure']
y = df['DiabetesPedigreeFunction']

#%%
# Pre-processing 
# Specify columns where you want to replace zero values (excluding 'Pregnancies', 'DiabetesPedigreeFunction', and 'Outcome')


# Columns to replace zero values

# Columns to replace zero values
columns_to_replace_zeros = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

# Replace zero values with NaN in the specified columns
for column_to_replace_zeros in columns_to_replace_zeros:
    zero_mask = df[column_to_replace_zeros] == 0
    df.loc[zero_mask, column_to_replace_zeros] = float('nan')

# Handle missing values (NaN) before applying KNN imputation
for column_to_replace_zeros in columns_to_replace_zeros:
    column_data = df[column_to_replace_zeros].values.reshape(-1, 1)
    imputer = KNNImputer(n_neighbors=1)
    df[column_to_replace_zeros] = imputer.fit_transform(column_data)

# Check if zero values are replaced
print("Imputed DataFrame:")
print(df.head())


#%%
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')
# Fit a linear regression model
model = LinearRegression()
model.fit(x.values.reshape(-1, 1), y)  # Reshape x to be a 2D array for fitting

# Predicted values
y_pred = model.predict(x.values.reshape(-1, 1))

# Plot the line of best fit
plt.plot(x, y_pred, color='yellow', linewidth=2, label='Line of Best Fit')

# Add labels and title
plt.xlabel('BloodPressure')
plt.ylabel('Diabetes Pedigree Function')
plt.title('Scatter Plot with Line of Best Fit')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
# %%
# Load the dataset
# Specify response and predictor variables(Making the Pairplot)
response_variable = 'Outcome'
predictor_variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'BMI', 'Age']

# Create pair plot using Seaborn
sns.pairplot(df, vars=predictor_variables, hue=response_variable, plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Predictor Variables by Response Variable (Outcome)')
plt.show()

# %%import pandas as pd

# Check for missing values in the entire DataFrame
missing_values = df.isna().sum()
print("Missing values per column:")
print(missing_values)

# %%


# Load your dataset into a Pandas DataFrame

# Specify the predictor variables (X) and the target variable (y)
X = df.drop('Outcome', axis=1)  # Predictor variables (all columns except 'Outcome')
y = df['Outcome']  # Target variable

# Split the data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
labels = sorted(y.unique())
# Check the class distribution in training and test sets
print("Training set class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest set class distribution:")
print(y_test.value_counts(normalize=True))
# %%

scaler = StandardScaler()

lr_models(X_train, X_test, y_train, y_test, labels, True)


# classifier_lr = LogisticRegression(max_iter=1000, random_state=42)
# classifier_kn = KNeighborsClassifier()
# classifier_rf = RandomForestClassifier()

# classifier_lr.fit(X_train, y_train)
# classifier_kn.fit(X_train, y_train)
# classifier_rf.fit(X_train, y_train)





# score_lr = classifier_lr.score(X_test, y_test)
# score_kn = classifier_kn.score(X_test, y_test)
# score_rf = classifier_rf.score(X_test, y_test)


# print("Logistic Regression Test Accuracy:", score_lr)
# print("K-Nearest Neighbors Test Accuracy:", score_kn)
# print("Random Forest Test Accuracy:", score_rf)
# %%

#Test for normality


#Remove the 'Outcome' column from the DataFrame
numerical_columns = df.drop('Outcome', axis=1).columns

# Perform Shapiro-Wilk test for normality for each numerical column
for column in numerical_columns:
    # Perform Shapiro-Wilk test
    statistic, p_value = shapiro(df[column])
    
    # Print the results
    print(f"Shapiro-Wilk test for column '{column}':")
    print(f"Test Statistic: {statistic:.4f}, P-value: {p_value:.4f}")
    
    # Interpret the results
    alpha = 0.05
    if p_value > alpha:
        print("   Sample looks normally distributed (fail to reject H0)")
    else:
        print("   Sample does not look normally distributed (reject H0)")
# %%
#DB Scan
numerical_columns = df.drop('Outcome', axis=1).columns

# Specify DBSCAN parameters
epsilon = 1  # Adjust epsilon based on your dataset
min_samples = 5  # Adjust min_samples based on your dataset

# Iterate over each numerical column and apply DBSCAN
for column in numerical_columns:
    # Extract column data as a NumPy array
    column_data = df[column].values.reshape(-1, 1)
    
    # Instantiate DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit DBSCAN to the column data
    dbscan.fit(column_data)
    
    # Extract labels and core sample indices
    labels = dbscan.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    # Print information about clusters and noise points
    print(f"DBSCAN Outlier Detection for column '{column}':")
    print(f"  Estimated number of clusters: {n_clusters_}")
    print(f"  Estimated number of noise points: {n_noise_}")
    
    # Print indices of outliers
    outlier_indices = np.where(labels == -1)[0]
    print("Outlier Indices:", outlier_indices)
    
    # Plot the clusters and outliers
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

    for column in numerical_columns:
    # Extract column data as a NumPy array
        column_data = df[column].values.reshape(-1, 1)
    
    # Instantiate DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # Fit DBSCAN to the column data
        dbscan.fit(column_data)
    
    # Extract labels and core sample indices
        labels = dbscan.labels_
    
    # Identify outliers
        outlier_indices = np.where(labels == -1)[0]
        print(f"Outliers for column '{column}':")
        for idx in outlier_indices:
            print(f"Index: {idx}, Value: {column_data[idx][0]}")
        print()
# %%
print(df.head)
# %%

# Sort the DataFrame by the 'Insulin' column in descending order
sorted_df = df.sort_values(by='Insulin', ascending=False)

# Display the top rows of the sorted DataFrame to see the highest values of 'Insulin'
print("Top values of Insulin column:")
print(sorted_df[['Insulin']].head(80))


# %%
df = pd.read_csv('diabetes.csv')

# X = df.loc[:, df.columns != 'Outcome'].values
y = df['Outcome'].values



columns_to_replace_zeros = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

# Replace zero values with NaN in the specified columns
for column_to_replace_zeros in columns_to_replace_zeros:
    zero_mask = df[column_to_replace_zeros] == 0
    df.loc[zero_mask, column_to_replace_zeros] = float('nan')

X = df.values

# Handle missing values (NaN) before applying KNN imputation

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

print(df[columns_to_replace_zeros].head(10))
print(X_imputed[:10, [2,3,4,5,7]])
# Check if zero values are replaced
# print("Imputed DataFrame:")
# print(df.head())
# %%
