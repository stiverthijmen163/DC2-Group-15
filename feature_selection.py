import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_importances(model_features: list) -> list:
    """
    Creates a list in descending order of feature names and corresponding feature values of a Decision Tree model.
    :param model_features: list of feature value of the used model
    :return: list of feature names and corresponding importance value in descending order
    """
    importances = pd.DataFrame(data={
        'attribute': X.columns,
        'importance': model_features
    })
    importances = importances.sort_values(by='importance', ascending=False)
    return importances


def plot_importances(importances: list, feature_amount: int, title: str):
    """
    Plots the top 20 feature importance values of a Decision Tree model.
    :param importances: list of the feature importance values
    :param feature_amount: amount of features to plot
    :param title: title for the plot
    """
    plt.bar(x=importances[:feature_amount]['attribute'], height=importances[:feature_amount]['importance'])
    plt.title(title)
    plt.xticks(rotation='vertical', size=5)
    plt.tight_layout()
    plt.show()


def print_importances(importances: list, title: str):
    """
    Prints the importance values of Decsion Tree model.
    :param importances: list of the feature importance values
    :param title: title to see what is printed
    """
    print(title)
    for index, row in importances.iterrows():
        attribute = row['attribute']
        score = row['importance']
        print(f'Attribute Name: {attribute}, Score: {score}')
    print('\n')


# Connect to db
cnx = sqlite3.connect('data/police_data.db')

# Group by measure since the trust is measured per measure
df_PAS_Borough = pd.read_sql_query("SELECT * FROM PAS_Borough", cnx)
df_stop_and_search = pd.read_sql_query("SELECT * FROM stop_and_search GROUP BY month, borough", cnx)

df_merged = pd.merge(df_PAS_Borough, df_stop_and_search, on=['borough', 'month'], how='inner')
df_merged = df_merged.drop(['survey', 'latitude', 'longitude', 'part_of_a_policing_operation',
                            'policing_operation', 'outcome_linked_to_object_of_search',
                            'removal_of_more_than_just_outer_clothing'], axis=1)

df_survey = pd.read_sql_query("SELECT * FROM PAS_questions", cnx)

# Inconsistencies in the data
df_survey['borough'] = df_survey['borough'].replace('kensington & chelsea', 'kensington and chelsea')
df_survey['borough'] = df_survey['borough'].replace('barking & dagenham', 'barking and dagenham')
df_survey['borough'] = df_survey['borough'].replace('hammersmith & fulham', 'hammersmith and fulham')
df_survey = df_survey.replace({'-': None, 'not asked': None})

# Feature analysis with Decision tree. NOTE: So far we can only use one variable, since other variables are strings
# To solve the string values we need to consider making dummy variables. For example: every crime types gets its own
# variable: 1 is yes, 0 is no.
# Source: https://machinelearningmastery.com/calculate-feature-importance-with-python/

# Converting the Boroughs to dummy variables
df_merged = pd.get_dummies(df_merged, columns=['type', 'gender', 'age_range',
                                                           'officer_defined_ethnicity', 'object_of_search', 'outcome'])

# Define X (the values we use to predict y) and define y (= trust level)
X, y = df_merged.iloc[:, 7:-1], df_merged['mps']

# Create and fit the model
model = DecisionTreeRegressor()   # Using random_state so that we can reproduce results, since the features are always randomly permuted at each split
model.fit(X, y)

# get importance
importances = get_importances(model.feature_importances_)

# print importances
print_importances(importances, 'Feature importance values using Decision Tree Regressor in descending order:')

# plot feature importance
plot_importances(importances, 20, "Top 20 feature importance using Decision Tree Regressor")

# Feature analysis with Random Forest Regression
model = RandomForestRegressor()

# fit the model
model.fit(X, y)

# get importance
importances = get_importances(model.feature_importances_)

# print importance
print_importances(importances, 'Feature importance values using Random Forest Regression in descending order:')

# plot feature importance
plot_importances(importances, 20, "Top 20 feature importance using Random Forest Regression")

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
importances = get_importances(model.feature_importances_)

# print importances
print_importances(importances, 'Feature importance values using XGBoost in descending order:')

# plot feature importance
plot_importances(importances, 20, "Top 20 feature importance using XGBoost")

# Things we still need to take into account: time lag!

# Decision tree model with PAS survey questions
df_questions = df_survey.loc[:, 'q1':]   # from question q1 till the end
answered_counts = df_questions.count()   # check how many times a question is answered to make a selection
additional_columns = df_survey.iloc[:, [0, 2]]   # want to include month and borough as well

# Sort the counts in descending order and get the top 20
top_20_answered_questions = answered_counts.sort_values(ascending=False).head(20).index   # only pick the 20 questions
df_top_questions = df_questions[top_20_answered_questions]
questions = df_top_questions.columns
df_top_questions = additional_columns.merge(df_top_questions, left_index=True, right_index=True)   # add month and borough to questions

# Create dummy variables and group by borough and month
df_top_questions = pd.get_dummies(df_top_questions, columns=questions)
df_top_questions = df_top_questions.groupby(["borough", "month"]).sum()   # or take mean value?

# Sum all previous three rows together independent of the rows before
df_top_questions = df_top_questions.rolling(3).sum()

# Only keep the quartile months so that we can join with the PAS_Borough
df_top_questions = df_top_questions.reset_index()
df_top_questions['month'] = pd.to_datetime(df_top_questions['month'])
df_top_questions = df_top_questions[df_top_questions['month'].dt.month.isin([3, 6, 9, 12])]

# Merge to two dataframes so that we have the trust together with the questions
df_PAS_Borough['month'] = pd.to_datetime(df_PAS_Borough['month'])
df_top_questions = pd.merge(df_top_questions, df_PAS_Borough[['borough', 'month', 'mps']].groupby(["borough", "month"]).mean(), on=['borough', 'month'])

# Define X (the values we use to predict y) and define y (= trust level)
X, y = df_top_questions.iloc[:, 2:-1], df_top_questions['mps']

# Create and fit the model
model = DecisionTreeRegressor()   # Consider using random_state parameter to recreate exact outcome
model.fit(X, y)

# get importance
importances = get_importances(model.feature_importances_)

# print importances
print_importances(importances,
                  'Feature importance values using Decision Tree Regressor in descending order (Survey Questions):')

# plot feature importance
plot_importances(importances, 20, "Top 20 feature importance using Decision Tree Regressor (Survey Questions)")

# Feature analysis with Random Forest Regression
model = RandomForestRegressor()

# fit the model
model.fit(X, y)

# get importance
importances = get_importances(model.feature_importances_)

# print importance
print_importances(importances,
                  'Feature importance values using Random Forest Regression in descending order (Survey Questions):')

# plot feature importance
plot_importances(importances, 20, "Top 20 feature importance using Random Forest Regression (Survey Questions)")

# Feature analysis with XGBoost
model = XGBRegressor()

# fit the model
model.fit(X, y)

# get importance
importances = get_importances(model.feature_importances_)

# print importances
print_importances(importances, 'Feature importance values using XGBoost in descending order (Survey Questions):')

# plot feature importance
plot_importances(importances, 20, "Top 20 feature importance using XGBoost (Survey Questions)")

# Feature analysis with Linear Regression
model = LinearRegression()

# fit the model
model.fit(X, y)

# get importance
importances = get_importances(model.coef_)

# print importances
print_importances(importances,
                  'Feature importance values using Linear Regression in descending order (Survey Questions):')

# plot feature importance
plot_importances(importances, -1, "Feature importance using Linear Regression (Survey Questions)")

# Plotting Decision Tree similar to the one in RegressionTree.py
# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
clf_dt = DecisionTreeRegressor(max_depth=3, random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, rounded=True, feature_names=X.columns)
plt.show()
