import pandas as pd
import sqlite3
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.model_selection import train_test_split
# from feature_selection import make_best_linear_regressions

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def make_best_linear_regressions(df, questions: list):
    """
    Plots the questions with the highest R^2 of the cleaned survey data set
    :param questions: list consisting the questions of the cleaned survey data set
    """
    df = df.copy()
    for question in questions:
        df['proportion_yes'] = df[question] / df['total respondents']
        df = df[df['proportion_yes'] > 0].copy()

        X = df[['proportion_yes']]
        y = df['proportion']

        # Create and fit the model
        try:
            model = LinearRegression()
            model.fit(X, y)
        except:
            continue

        # Create predictions and get R^2
        y_pred = model.predict(X)
        R_squared = model.score(X, y)
        # Plot if R-squared is in range
        if R_squared > 0.70 and len(df) > 3:
            print(question)
            print(X)
            plt.scatter(X, y, color='blue', label='Original data')
            plt.plot(X, y_pred, color='red', label='Regression line')
            plt.xlabel(f'Proportion of persons that answered {question}')
            plt.ylabel('MPS')
            plt.title(f'Linear Regression of MPS and question {question}')
            plt.legend()
            plt.show()


if __name__ == "__main__":
    conn = sqlite3.connect("data/police_data.db")
    conn_clean = sqlite3.connect("data/cleaned_police_data.db")

    query = """SELECT * FROM PAS_questions_cleaned"""
    df = pd.read_sql_query(query, conn_clean)
    print(df)

    query = """SELECT borough, month, AVG(proportion) AS proportion
    FROM PAS_Borough
    GROUP BY borough, month"""
    df_borough = pd.read_sql_query(query, conn)

    df = pd.merge(df, df_borough, on=["borough", "month"])
    print(df)
    df['total respondents'] = df.iloc[:, 2:10].sum(axis=1)

    column_sums = df[df.columns[2:]].sum()
    columns_to_drop = column_sums[column_sums < 1000].index

    if "proportion" in columns_to_drop:
        print(columns_to_drop)
        # print("hi")
        columns_to_drop = columns_to_drop.difference(["proportion"])
        print(columns_to_drop)
    df = df.drop(columns=columns_to_drop)
    # print(df)
    # print(df.columns.to_list()[2:])

    make_best_linear_regressions(df, df.columns[2:])
