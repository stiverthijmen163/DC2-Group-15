import sqlite3
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

def make_linear_regression(question: str, show: bool):
    """
    Creates and plots a regression model that predicts the MPS by one question of the df_survey_cleaned dataframe.
    :param question: dummy variable name from PAS survey that is a question with corresponding answer (1 = yes, 0 = no)
    :param show: whether to show the linear regression plot
    """
    df_survey_cleaned['proportion_yes'] = df_survey_cleaned[question] / df_survey_cleaned['total']
    df = df_survey_cleaned[df_survey_cleaned['proportion_yes'] > 0]
    X = df[['proportion_yes']]
    y = df[['proportion']]

    # Create model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions
    y_pred = model.predict(X)

    if show:
        # Plot the original data points
        plt.scatter(X, y, color='blue', label='Original data')

        # Plot the regression line
        plt.plot(X, y_pred, color='red', label='Regression line')

        # Showing regression plot
        # Add labels and legend
        plt.xlabel(f'Proportion of persons that answered {question}')
        plt.ylabel('Confidence')
        plt.title(f'Linear Regression of Confidence and question {question}')
        plt.legend()
        plt.show()

def make_best_linear_regressions(questions: list):
    """
    Plots the questions with the highest R^2 of the cleaned survey data set
    :param questions: list consisting the questions of the cleaned survey data set
    """
    for question in questions:
        df_survey_cleaned['proportion_yes'] = df_survey_cleaned[question] / df_survey_cleaned['total']
        df = df_survey_cleaned[df_survey_cleaned['proportion_yes'] > 0]

        X = df[['proportion_yes']]
        y = df['proportion']

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Create predictions and get R^2
        y_pred = model.predict(X)
        R_squared = model.score(X, y)
        # Plot if R-squared is in range
        if R_squared > 0.70:
            print(question)
            plt.scatter(X, y, color='blue', label='Original data')
            plt.plot(X, y_pred, color='red', label='Regression line')
            plt.xlabel(f'Proportion of persons that answered {question}')
            plt.ylabel('Confidence')
            plt.title(f'Linear Regression of Confidence and question {question}')
            plt.legend()
            plt.show()


def make_more_linear_regressions(questions, rows, columns, filename):
    """
    Creates and plots regression models in a grid that predict the MPS by each question in the list of questions.
    :param questions: list of dummy variable names from PAS survey that are questions with corresponding answers (1 = yes, 0 = no)
    :param rows: number of rows for the plot grid
    :param columns: number of columns for the plot grid
    :param filename: name of the file to save the plot
    """
    # Create a 3x2 grid of subplots
    fig, axs = plt.subplots(rows, columns, figsize=(15, 15))

    # Flatten the array of axes for easier iteration
    axs = axs.flatten()

    # Plot each question on a separate subplot
    for ax, question in zip(axs, questions):
        df_survey_cleaned['proportion_yes'] = df_survey_cleaned[question] / df_survey_cleaned['total']
        df = df_survey_cleaned[df_survey_cleaned['proportion_yes'] > 0]
        X = df[['proportion_yes']]
        y = df[['proportion']]

        # Create model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Plot original
        ax.scatter(X, y, color='blue', label='Original data')

        # Plot the regression line
        ax.plot(X, y_pred, color='red', label='Regression line')

        # Add labels and legend
        ax.set_xlabel(f'Proportion of persons that answered {question}')
        ax.set_ylabel('Confidence')
        ax.set_title(f'Linear Regression of Confidence and question {question}')
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Shave the plot
    plt.savefig(filename, type='svg')


if __name__ == "__main__":
    # Connect to db
    cnx = sqlite3.connect('data/police_data.db')
    cnx_cleaned = sqlite3.connect('data/cleaned_police_data.db')

    # Group by measure since the trust is measured per measure
    df_PAS_Borough = pd.read_sql_query("""SELECT * FROM PAS_Borough WHERE measure = '"good job" local'""", cnx)
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
    X, y = df_merged.iloc[:, 7:-1], df_merged['proportion']

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
                                df_PAS_Borough[['borough', 'month', 'proportion']].groupby(["borough", "month"]).mean(),
                                on=['borough', 'month'])

    # Define X (the values we use to predict y) and define y (= trust level)
    X, y = df_top_questions.iloc[:, 2:-1], df_top_questions['proportion']
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
    df_survey_cleaned['month'] = pd.to_datetime(df_survey_cleaned['month'])   # convert to datetime so that we can join two datetime columns with each other

    # Get total number of respondents (since question 1 is always asked, we can sum the values of the dummy variables of
    # question 1 to get the total number of respondents)
    df_survey_cleaned['total'] = df_survey_cleaned.iloc[:, 2:10].sum(axis=1)

    df_survey_cleaned.drop(columns=['borough'],
                           inplace=True)   # dropping borough column since MPS is the same for every borough

    # Want to exclude all questions (= all except month and borough column) that have been answered less than 1000 times, to
    # avoid overfitting
    column_sums = df_survey_cleaned[df_survey_cleaned.columns[2:]].sum()
    columns_to_drop = column_sums[column_sums < 1000].index
    df_survey_cleaned = df_survey_cleaned.drop(columns=columns_to_drop)

    # Grouping and joining so that we can use MPS
    df_survey_cleaned = pd.merge(df_survey_cleaned.groupby(['month']).sum(),
                                 df_PAS_Borough[['month', 'proportion']].groupby(['month']).mean(),
                                 on=['month'])

    # Question q39a_2: To what extent do you think knife crime is a problem in this area? By knife crime I mean people
    # carrying or using knives to threaten or commit violence.
    make_linear_regression('q39a_2_not a problem at all', True)
    make_linear_regression('q39a_2_major problem', True)

    # Question nq133a: Do you know how to contact your Safer Neighbourhood Team or your Dedicated Ward Officers?
    # If asked: You can find out more about your local team by entering your postcode or looking up
    # your borough on the website http://www.met.police.uk/saferneighbourhoods/.
    make_linear_regression('nq133a_yes', True)

    # Question q61: Taking everything into account, how good a job do you think the police IN LONDON AS A WHOLE
    # are doing?
    make_linear_regression('q61_good', True)
    make_linear_regression('q61_fair', False)
    make_linear_regression('q61_poor', True)

    # Question q62a: To what extent do you agree with these statements about the police in your area? By 'your area' I
    # mean within 15 minutes' walk from your home.
    # They can be relied on to be there when you need them
    make_linear_regression('q62a_tend to agree', True)

    # Question a120: ‘Stop and Search’ is a power that allows the police to speak to someone if they think they have
    # been involved in a crime, and to search them to see whether they are carrying anything that they
    # should not be.
    # To what extent do you agree that the Police should conduct Stop and Search?
    make_linear_regression('a120_strongly agree', True)

    # Question rq80e: Your Safer Neighbourhood Team is a group of police officers dedicated to serving your community.
    # The team includes 2 officers (Dedicated Ward Officers) based in your area (or 'ward'), supported
    # by additional officers from the wider area.
    # Prior to this interview, had you heard about your Safer Neighbourhood Team or your Dedicated
    # Ward Officers?
    make_linear_regression('rq80e_no', False)
    make_linear_regression('rq80e_yes', False)

    # Question nq147r: What is your ethnic group?
    # NOTE: this question was not always asked, so beware of outliers.
    make_linear_regression('nq147r_black', False)
    make_linear_regression('nq147r_asian', False)
    make_linear_regression('nq147r_white british', False)

    # Question q150r: What is your sexual orientation?
    # INTERVIEWER: Read out options only if necessary.
    # If necessary remind the respondent that they do not need to answer if they would prefer not to
    # say.
    make_linear_regression('q150r_heterosexual', False)
    make_linear_regression('q150r_non-heterosexual', False)

    # Question nq149r: What is your religion, even if you are not currently practicing?
    make_linear_regression('nq149r_christian', True)
    make_linear_regression('nq149r_muslim', False)
    make_linear_regression('nq149r_hindu', True)
    make_linear_regression('nq149r_no religion', False)

    # Question xq135: What is your sex?
    make_linear_regression('xq135r_male', True)
    make_linear_regression('xq135r_female', True)

    # Question nq135bd: To what extent do you agree or disagree with the following statements:
    # The Metropolitan Police Service is an organisation that I can trust
    make_linear_regression('nq135bd_strongly agree', False)
    make_linear_regression('nq135bd_neither agree nor disagree', True)
    make_linear_regression('nq135bd_strongly disagree', False)

    # Question q15: To what extent are you worried about…
    # Anti-social behaviour in your area?
    # IF necessary: By this I mean issues such as vandalism, using or dealing drugs, people being drunk
    # or rowdy, teenagers hanging around on the streets, or noisy neighbours?
    make_linear_regression('q15_very worried', True)
    make_linear_regression('q15_not at all worried', True)

    # Question q65:
    # On average, how often do YOU see the police PATROLLING ON FOOT, BICYCLE OR HORSEBACK IN
    # THIS AREA? Remember I am talking about the area within 15 minutes’ walk from here.
    # If necessary: This does include PSCOs and we are talking about how often they “currently” see
    # them.
    make_linear_regression('q65_at least daily', True)
    make_linear_regression('q65_never', True)

    # Question q79g:
    # Please use a scale of 1 to 7, where 1 = Not at all well and 7 = Very well
    # How well do you think the Metropolitan Police… Tackle drug dealing and drug use?
    # If necessary: Please think of London as a whole, rather than your local area in this instance
    make_linear_regression('q79g_1 not at all well', True)

    # Question q79d:
    # …Tackle gun crime? (scale 1 to 7)
    make_linear_regression('q79d_2', True)

    # Question q79b:
    # …Respond to emergencies promptly? (scale 1 to 7)
    make_linear_regression('q79b_3', False)

    # Question 62f:
    # To what extent do you agree with these statements about the police in your area? By 'your area' I
    # mean within 15 minutes' walk from your home.
    # They are dealing with the things that matter to people in this community
    make_linear_regression('q62f_tend to disagree', False)

    # Question q66:
    # On average, how often do YOU see the police PATROLLING ON FOOT OR BICYCLE IN THIS AREA? Remember I am talking about
    # the area within 15 minutes' walk. Do you think this is...?
    make_linear_regression('q66_not often enough', False)

    # Get regression plots with highest R^2
    questions = df_survey_cleaned.columns[:-3]
    make_best_linear_regressions(questions)

    # plot subplots for findings section in paper
    report_questions = ['a120_strongly agree', 'q66_not often enough', 'q79g_1 not at all well', 'q15_very worried',
                        'q15_not at all worried', 'q62f_tend to disagree']
    #make_more_linear_regressions(report_questions, 3, 2, 'report_findings_LR')

