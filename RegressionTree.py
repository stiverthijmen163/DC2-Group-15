import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Fix matplotlib
matplotlib.use('TkAgg')

# Load an Excel file
excel_file_path = "C:/Users/20223925/Desktop/data/PAS_MPS.xlsx"

# Load data from the excel file
PAS_MPS = pd.read_excel(excel_file_path, index_col=[0])

# Remove rows with any NaN values
PAS_MPS = PAS_MPS.dropna()

# Create prediction variables and predictor
X = PAS_MPS.drop(['Trust MPS', '"Good Job" local'], axis=1).copy()
y = PAS_MPS['Trust MPS'].copy()

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
clf_dt = DecisionTreeRegressor(random_state=42, ccp_alpha=0.00005)
clf_dt = clf_dt.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, rounded=True, feature_names=X.columns)
plt.show()
