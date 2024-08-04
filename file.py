
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

from modeling import lr_models, rf_model, kn_model, knn_fold, rf_fold

from preprocessing import clus_detection, imputing, normality

#%%
df = pd.read_csv('diabetes.csv')

x_real = df.drop('Outcome', axis=1)
y_real = df['Outcome']

#%%
# Imputer Function
imputed_df = imputing(5, df)

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


# lr_models(X_train, X_test, y_train, y_test, labels, True)

# rf_model(X_train, X_test, y_train, y_test, labels, True)

# kn_model(X_train, X_test, y_train, y_test, labels, True)

best_k, test_loss_kn = knn_fold(1, 30, 10, imputed_df, y_real)

# best_ne, best_md, test_loss_rf = 
# rf_fold(50, 100, 10, 50, 10, imputed_df, y_real)


# print("Best k for KNN:", best_k)
# print("Test loss for KNN:", test_loss_kn)
# print("Best number of estimators for RF:", best_ne)
# print("Best max depth for RF:", best_md)
# print("Test loss for RF:", test_loss_rf)





# %%
#Test for normality
normality(df,0.05)
# %%
#DB Scan
clus_detection(1, 5, df, True)


# %%
# df = pd.read_csv('diabetes.csv')

# # X = df.loc[:, df.columns != 'Outcome'].values
# y = df['Outcome'].values



# columns_to_replace_zeros = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

# # Replace zero values with NaN in the specified columns
# for column_to_replace_zeros in columns_to_replace_zeros:
#     zero_mask = df[column_to_replace_zeros] == 0
#     df.loc[zero_mask, column_to_replace_zeros] = float('nan')

# X = df.values

# # Handle missing values (NaN) before applying KNN imputation

# imputer = KNNImputer(n_neighbors=5)
# X_imputed = imputer.fit_transform(X)

# print(df[columns_to_replace_zeros].head(10))
# print(X_imputed[:10, [2,3,4,5,7]])
# # Check if zero values are replaced
# # print("Imputed DataFrame:")
# # print(df.head())
