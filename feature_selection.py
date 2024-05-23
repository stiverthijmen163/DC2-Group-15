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
    Prints the top 20 importance values of Decsion Tree model.
    :param importances: list of the feature importance values
    :param title: title to see what is printed
    """
    print(title)
    for index, row in importances[:20].iterrows():
        attribute = row['attribute']
        score = row['importance']
        print(f'Attribute Name: {attribute}, Score: {score}')
    print('\n')

def make_linear_regression(question: str):
    """
    Creates and plots a regression model that predicts the MPS by one question of the df_survey_cleaned dataframe.
    :param question: dummy variable name from PAS survey that is a question with corresponding answer (1 = yes, 0 = no)
    """
    # THIS IS NOT HOW YOU CALCULATE PROPORTIONS: THE SAME PERSON MIGHT HAVE ANSWERED MORE THAN ONE QUESTIONS!
    df_survey_cleaned['proportion_yes'] = df_survey_cleaned[question] / df_survey_cleaned['total respondents']
    df = df_survey_cleaned[df_survey_cleaned['proportion_yes'] > 0]
    X = df[['proportion_yes']]
    y = df[['mps']]

    # Create model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions
    y_pred = model.predict(X)

    # Plot the original data points
    plt.scatter(X, y, color='blue', label='Original data')

    # Plot the regression line
    plt.plot(X, y_pred, color='red', label='Regression line')

    # Add labels and legend
    plt.xlabel(f'Proportion of persons that answered {question}')
    plt.ylabel('MPS')
    plt.title(f'Linear Regression of MPS and question {question}')
    plt.legend()
    plt.show()


# Connect to db
cnx = sqlite3.connect('data/police_data.db')
cnx_cleaned = sqlite3.connect('data/cleaned_police_data.db')

# Group by measure since the trust is measured per measure
df_PAS_Borough = pd.read_sql_query("SELECT * FROM PAS_Borough", cnx)
df_stop_and_search = pd.read_sql_query("SELECT * FROM stop_and_search GROUP BY month, borough", cnx)

df_merged = pd.merge(df_PAS_Borough, df_stop_and_search, on=['borough', 'month'], how='inner')
df_merged = df_merged.drop(['survey', 'latitude', 'longitude', 'part_of_a_policing_operation',
                            'policing_operation', 'outcome_linked_to_object_of_search',
                            'removal_of_more_than_just_outer_clothing'], axis=1)

df_survey = pd.read_sql_query("SELECT * FROM PAS_questions", cnx)
df_survey_cleaned = pd.read_sql_query("SELECT * FROM PAS_questions_cleaned", cnx_cleaned)

# Inconsistencies in the data
df_survey['borough'] = df_survey['borough'].replace('kensington & chelsea', 'kensington and chelsea')
df_survey['borough'] = df_survey['borough'].replace('barking & dagenham', 'barking and dagenham')
df_survey['borough'] = df_survey['borough'].replace('hammersmith & fulham', 'hammersmith and fulham')
df_survey = df_survey.replace({'-': None, 'not asked': None})
df_survey = df_survey.dropna(axis=1, thresh=int(len(df_survey) * 0.90))

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
top_20_answered_questions = answered_counts.sort_values(ascending=False).index   # only pick the 20 questions
df_top_questions = df_questions[top_20_answered_questions]
questions = df_top_questions.columns
df_top_questions = additional_columns.merge(df_top_questions,
                                            left_index=True, right_index=True)   # add month and borough to questions

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
df_top_questions = pd.merge(df_top_questions,
                            df_PAS_Borough[['borough', 'month', 'mps']].groupby(["borough", "month"]).mean(),
                            on=['borough', 'month'])

# Define X (the values we use to predict y) and define y (= trust level)
X, y = df_top_questions.iloc[:, 2:-1], df_top_questions['mps']
# TO DO: only keep columns that have a sum value of >= x
counts = X.sum()
filtered_columns = counts[counts > 10000].index   # Questions that have been answered at least x times to prevent overfitting
X = X[filtered_columns]

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
clf_dt = DecisionTreeRegressor(max_depth=2, random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, filled=True, rounded=True, feature_names=X.columns)
plt.show()

# BEWARE OF INCORRECT RESULTS! Some questions are only asked in specific years, such as the body cam questions.
# If a question is not asked in other years, it will have a value of 0 in the dummy DataFrame. Even replacing this value
# with None won't help, as the decision tree will interpret it as a 0 as well.
# This leads to a problem: for example, if the body cam question is only asked in 2015 and not in subsequent years, it
# will have only 0 (or None) values for the other dates. Since the MPS (trust level) decreases over time, the decision
# tree model mistakenly interprets these 0 values as being correlated with or even causing the lower trust. As a result,
# the model assigns high importance to this question, thinking it is a good predictor, when in reality, it is not.
# Now there are two solutions for this: we could either only look at the questions that were asked over the whole
# timeline, or we could group by timeline, so that we every question is in a dataframe with their corresponding
# timeline. For now, I used the first solution.

# Create Linear Regression model to check decision tree results
df_survey_cleaned['month'] = pd.to_datetime(
    df_survey_cleaned['month'])   # convert to datetime so that we can join two datetime columns with each other

# Get total number of respondents (since question 1 is always asked, we can sum the values of the dummy variables of
# question 1 to get the total number of respondents)
df_survey_cleaned['total respondents'] = df_survey_cleaned.iloc[:, 2:10].sum(axis=1)

df_survey_cleaned.drop(columns=['borough'],  # CALCULATE PROPORTION HERE!!
                       inplace=True)   # dropping borough column since MPS is the same for every borough

# Grouping and joining so that we can use MPS
df_survey_cleaned = pd.merge(df_survey_cleaned.groupby(['month']).sum(),
                             df_PAS_Borough[['month', 'mps']].groupby(['month']).mean(),
                             on=['month'])


# Question q39a_2: To what extent do you think knife crime is a problem in this area? By knife crime I mean people
# carrying or using knives to threaten or commit violence.
make_linear_regression('q39a_2_not a problem at all')

# Question nq133a: Do you know how to contact your Safer Neighbourhood Team or your Dedicated Ward Officers?
# If asked: You can find out more about your local team by entering your postcode or looking up
# your borough on the website http://www.met.police.uk/saferneighbourhoods/.
make_linear_regression('nq133a_yes')

# Question q61: Taking everything into account, how good a job do you think the police IN LONDON AS A WHOLE
# are doing?
make_linear_regression('q61_good')
make_linear_regression('q61_fair')
make_linear_regression('q61_poor')

# Question q62a: To what extent do you agree with these statements about the police in your area? By 'your area' I
# mean within 15 minutes' walk from your home.
# They can be relied on to be there when you need them
make_linear_regression('q62a_tend to agree')

# Question a120: ‘Stop and Search’ is a power that allows the police to speak to someone if they think they have
# been involved in a crime, and to search them to see whether they are carrying anything that they
# should not be.
# To what extent do you agree that the Police should conduct Stop and Search?
make_linear_regression('a120_strongly agree')

# Question rq80e: Your Safer Neighbourhood Team is a group of police officers dedicated to serving your community.
# The team includes 2 officers (Dedicated Ward Officers) based in your area (or 'ward'), supported
# by additional officers from the wider area.
# Prior to this interview, had you heard about your Safer Neighbourhood Team or your Dedicated
# Ward Officers?
make_linear_regression('rq80e_no')

# Question nq147r: What is your ethnic group?
# NOTE: this question was not always asked, so beware of outliers.
make_linear_regression('nq147r_black')
make_linear_regression('nq147r_asian')
make_linear_regression('nq147r_white british')

# Question q150r: What is your sexual orientation?
# INTERVIEWER: Read out options only if necessary.
# If necessary remind the respondent that they do not need to answer if they would prefer not to
# say.
make_linear_regression('q150r_heterosexual')
make_linear_regression('q150r_non-heterosexual')

# Question nq149r: What is your religion, even if you are not currently practicing?
make_linear_regression('nq149r_christian')
make_linear_regression('nq149r_muslim')
make_linear_regression('nq149r_hindu')
make_linear_regression('nq149r_no religion')

# Question xq135: What is your sex?
make_linear_regression('xq135r_male')
make_linear_regression('xq135r_female')

# Question nq135bd: To what extent do you agree or disagree with the following statements:
# The Metropolitan Police Service is an organisation that I can trust
make_linear_regression('nq135bd_strongly agree')
make_linear_regression('nq135bd_neither agree nor disagree')
make_linear_regression('nq135bd_strongly disagree')

# FOR TOTAL RESPONDANTS: SUM THE VALUE OF THE DUMMY VARIABLES OF QUESTION 1
