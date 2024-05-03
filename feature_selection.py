import sqlite3
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot


cnx = sqlite3.connect('data/data/police_data.db')
df_PAS_Borough = pd.read_sql_query("SELECT * FROM PAS_Borough GROUP BY Date, Borough, Measure", cnx)

# Inconsistency in the data: some rows have upon without capital U, others do
df_PAS_Borough['Borough'] = df_PAS_Borough['Borough'].replace('Richmond Upon Thames', 'Richmond upon Thames')

# Feature analysis with Decision tree. NOTE: So far we can only use one variable, since other variables are strings
# To solve the string values we need to consider making dummy variables. For example: every crime types gets its own
# variable: 1 is yes, 0 is no.
# Source: https://machinelearningmastery.com/calculate-feature-importance-with-python/

# Converting the Boroughs to dummy variables
df_PAS_Borough = pd.get_dummies(df_PAS_Borough, columns=['Borough'])

print(df_PAS_Borough.iloc[:, 5:-1].columns)

# Define X (the values we use to predict y) and define Y (= trust level)
X, y = df_PAS_Borough.iloc[:, 5:-1], df_PAS_Borough['MPS']

# Create and fit the model
model = DecisionTreeRegressor()
model.fit(X, y)

# get importance
importance = model.feature_importances_

# summarize feature importance
for i in range(len(importance)):
    print(f'Feature {i+1}: {df_PAS_Borough.iloc[:, 5:-1].columns[i]}, Score: {importance[i]}')

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

