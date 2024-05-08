import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Connect to db
cnx = sqlite3.connect('data/police_data.db')

# Group by measure since the trust is measured per measure
df_PAS_Borough = pd.read_sql_query("SELECT * FROM PAS_Borough", cnx)
df_stop_and_search = pd.read_sql_query("SELECT * FROM stop_and_search GROUP BY month, borough", cnx)

df_merged = pd.merge(df_PAS_Borough, df_stop_and_search, on=['borough', 'month'], how='inner')
df_merged = df_merged.drop(['survey', 'latitude', 'longitude', 'part_of_a_policing_operation',
                            'policing_operation', 'outcome_linked_to_object_of_search',
                            'removal_of_more_than_just_outer_clothing'], axis=1)


# Feature analysis with Decision tree. NOTE: So far we can only use one variable, since other variables are strings
# To solve the string values we need to consider making dummy variables. For example: every crime types gets its own
# variable: 1 is yes, 0 is no.
# Source: https://machinelearningmastery.com/calculate-feature-importance-with-python/

# Converting the Boroughs to dummy variables
df_merged = df_merged = pd.get_dummies(df_merged, columns=['type', 'gender', 'age_range',
                                                           'officer_defined_ethnicity', 'object_of_search', 'outcome'])

# Define X (the values we use to predict y) and define y (= trust level)
X, y = df_merged.iloc[:, 7:-1], df_merged['mps']

# Create and fit the model
model = DecisionTreeRegressor()   # Using random_state so that we can reproduce results, since the features are always randomly permuted at each split
model.fit(X, y)

# get importance
importances = pd.DataFrame(data={
    'attribute': X.columns,
    'importance': model.feature_importances_
})
importances = importances.sort_values(by='importance', ascending=False)

print('Feature importance values using Decision Tree Regressor in descending order:')
for index, row in importances.iterrows():
    attribute = row['attribute']
    score = row['importance']
    print(f'Attribute Name: {attribute}, Score: {score}')


# plot feature importance
plt.bar(x=importances['attribute'], height=importances['importance'])
plt.title("Feature importance using Decision Tree Regressor")
plt.xticks(rotation='vertical', size=5)
plt.show()

# Feature analysis with Random Forest Regression
model = RandomForestRegressor()

# fit the model
model.fit(X, y)

# get importance
importances = pd.DataFrame(data={
    'attribute': X.columns,
    'importance': model.feature_importances_
})
importances = importances.sort_values(by='importance', ascending=False)

# summarize feature importance
print('\n')
print('Feature importance values using Random Forest Regression in descending order:')
for index, row in importances.iterrows():
    attribute = row['attribute']
    score = row['importance']
    print(f'Attribute Name: {attribute}, Score: {score}')

# plot feature importance
plt.bar(x=importances['attribute'], height=importances['importance'])
plt.title("Feature importance using Random Forest Regression")
plt.xticks(rotation='vertical', size=5)
plt.show()

# Feature analysis with XGBoost (Extreme Gradient Boosting)
# XGBoost is based on the concept of gradient boosting, which is an ensemble technique where multiple weak learners
# (typically decision trees) are combined to create a strong learner. Gradient boosting builds trees sequentially,
# where each new tree tries to correct the errors made by the previous ones.
# documnetation link: https://xgboost.readthedocs.io/en/latest/index.html

# create model
model = XGBRegressor()

# fit the model
model.fit(X, y)

# get importance
importances = pd.DataFrame(data={
    'attribute': X.columns,
    'importance': model.feature_importances_
})
importances = importances.sort_values(by='importance', ascending=False)

# summarize feature importance
print('\n')
print('Feature importance values using XGBoost in descending order:')
for index, row in importances.iterrows():
    attribute = row['attribute']
    score = row['importance']
    print(f'Attribute Name: {attribute}, Score: {score}')

# plot feature importance
plt.bar(x=importances['attribute'], height=importances['importance'])
plt.title("Feature importance using XGBoost")
plt.xticks(rotation='vertical', size=5)
plt.show()

# Things we still need to take into account: time lag!
