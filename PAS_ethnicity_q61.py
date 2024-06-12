import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.model_selection import train_test_split
# from feature_selection import make_best_linear_regressions

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

def make_best_linear_regressions(df: pd.DataFrame, df_a: pd.DataFrame, df_r: pd.DataFrame, questions: list, borough: str, t_or_c: str):
    """
    Plots the questions with the highest R^2 of the cleaned survey data set
    :param questions: list consisting the questions of the cleaned survey data set
    """
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    combined_columns = list(set(df.columns.to_list()) & set(df_a.columns.to_list()) & set(df_r.columns.to_list()))
    ages = df_a["q136r"].unique().tolist()
    races = df_r["nq147r"].unique().tolist()
    # print(type(ages))
    # print(races)
    # df0 = df.copy()
    for question in questions:
        if question in combined_columns:
            show = False
            show_race = False
            show_age = False
            df0 = df.copy()
            df0['proportion_yes'] = df0[question] / df0['total']
            df0 = df0[df0['proportion_yes'] > 0].copy()
            X = df0[['proportion_yes']]
            y = df0['proportion']

            # Create and fit the model
            try:
                model = LinearRegression()
                model.fit(X, y)
            except:
                continue

            # Create predictions and get R^2
            y_pred = model.predict(X)
            R_squared = model.score(X, y)

            plt.figure(figsize=(12, 18))
            ax1 = plt.subplot(3, 1, 1)

            # Plot if R-squared is in range
            if R_squared > 0.70 and len(df0) > 3:
                show = True
            # if not show:
            #     continue
            #     print(question)
            #     print(df)
            #     print(X)
                plt.scatter(X, y, color='blue', label='Original data')
                plt.plot(X, y_pred, color='red', label='Regression line')
            #     plt.xlabel(f'Proportion of persons that answered {question}')
            #     plt.ylabel('MPS')
            #     plt.title(f'Linear Regression of MPS and question {question}')
            #     plt.legend()
            #     plt.savefig(f"artifacts/{borough}_{question}.png")
            #     plt.show()
            else:
                plt.scatter(X, y, color='blue', label='Original data', alpha=0.05)
                plt.plot(X, y_pred, label=f'Regression line {R_squared:.2f}, {len(df0)}', color="red", alpha=0.05)

            plt.xlabel(f'Proportion of persons that answered {question}')
            plt.ylabel(t_or_c.capitalize())
            plt.title(f'Linear Regression of {t_or_c} and question {question}')
            plt.legend()

            age_count = 0

            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            for age in ages:
                df_an = df_a[df_a["q136r"] == age].copy()
                # print(df_an)

                df_an['proportion_yes'] = df_an[question] / df_an['total']
                df_an = df_an[df_an['proportion_yes'] > 0].copy()
                X = df_an[['proportion_yes']]
                y = df_an['q61']

                # Create and fit the model
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    # print("OK")
                except:
                    # print(" NOT OK")
                    # print(df_a)
                    continue

                # Create predictions and get R^2
                y_pred = model.predict(X)
                R_squared = model.score(X, y)

                # ax2 = plt.subplot(3, 1, 2, sharex=ax1)

                if len(df_an) > 3 and R_squared > 0.7:
                    # plt.plot(X, y_pred, label=f'Regression line {age}, {R_squared:.2f}')

                    # if R_squared > 0.7:
                    plt.scatter(X, y, label='Original data', color=colors[ages.index(age)])
                    plt.plot(X, y_pred, label=f'Regression line {age}, {R_squared:.2f}, {len(df_an)}', color=colors[ages.index(age)])
                    age_count += 1
                else:
                    plt.plot(X, y_pred, label=f'Regression line {age}, {R_squared:.2f}, {len(df_an)}', color=colors[ages.index(age)], alpha=0.05)

                if age_count > 1:
                    show_age = True
            #
            # if show:
                plt.xlabel(f'Proportion of persons that answered {question}')
                plt.ylabel('Average answer to q61 (0 - very poor up to 1 - excellent)')
                plt.title(f'Linear Regression of q61 and question {question}')
                plt.legend()
            #     plt.show()
            # plt.close("all")

            race_count = 0

            for race in races:
                df_rn = df_r[df_r["nq147r"] == race].copy()
                # print(df_an)

                df_rn['proportion_yes'] = df_rn[question] / df_rn['total']
                df_rn = df_rn[df_rn['proportion_yes'] > 0].copy()
                X = df_rn[['proportion_yes']]
                y = df_rn['q61']

                # Create and fit the model
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    # print("OK")
                except:
                    # print(" NOT OK")
                    # print(df_a)
                    continue

                # Create predictions and get R^2
                y_pred = model.predict(X)
                R_squared = model.score(X, y)


                ax3 = plt.subplot(3, 1, 3, sharex=ax1, sharey=ax2)

                if len(df_rn) > 3 and R_squared > 0.7:
                    # plt.plot(X, y_pred, label=f'Regression line {age}, {R_squared:.2f}')

                    # if R_squared > 0.7:
                    plt.scatter(X, y, label='Original data', color=colors[races.index(race)])
                    plt.plot(X, y_pred, label=f'Regression line {race}, {R_squared:.2f}, {len(df_rn)}', color=colors[races.index(race)])
                    # show = True
                    race_count += 1
                else:
                    plt.plot(X, y_pred, label=f'Regression line {race}, {R_squared:.2f}, {len(df_rn)}', color=colors[races.index(race)], alpha=0.05)

                if race_count > 1:
                    show_race = True

            if show or show_race or show_age:
                plt.xlabel(f'Proportion of persons that answered {question}')
                plt.ylabel('Average answer to q61 (0 - very poor up to 1 - excellent)')
                plt.title(f'Linear Regression of q61 and question {question}')
                plt.suptitle(f"{t_or_c} in {borough} for question {question}\n", weight='bold')
                plt.legend()
                # plt.show()
                # plt.figure(figsize=(12, 18))
                question = question.replace('/', ' or ')
                plt.tight_layout()
                plt.savefig(f"artifacts/{t_or_c}/{borough}_{question}.png")
                # plt.figure(figsize=(12, 36))
            plt.close("all")


def month_to_quartile(month: str) -> str:
    """
    Convert a date in format 'YYYY/MM' to quartiles formatted as 'YYYY/MM'
    :param month: date in format 'YYYY/MM' to convert to quartile
    :return: quartiles formatted as 'YYYY/MM'
    """
    month_c = int(month[5:7])
    if month_c <= 3:
        q = "03"
    elif month_c <= 6:
        q = "06"
    elif month_c <= 9:
        q = "09"
    else:
        q = "12"

    return f"{month[:4]}-{q}"


def full_process_lin_reg_boroughs(trust_or_conf: bool):
    conn = sqlite3.connect("data/police_data.db")
    conn_clean = sqlite3.connect("data/cleaned_police_data.db")

    if trust_or_conf:  # trust
        query = """SELECT borough, month, AVG(proportion) AS proportion
        FROM PAS_Borough
        where measure == 'trust mps'
        GROUP BY borough, month"""

        t_or_c = "trust"
        q = ["q3k_strongly disagree", "q13_very worried", "q15_very worried", "q39a_2_major problem", "q39a_2_minor problem", "q39a_2_not a problem at all", "nq43_major problem", "nq43_not a problem at all", "q60_good", "q60_very poor", "q61_good", "q61_poor", "q61_very poor", "q62a_strongly disagree", "q62a_tend to agree", "q66_about right", "q66_not often enough", "q79b_3", "q79d_1 not at all well", "q79d_2", "q79g_1 not at all well", "q79j_don't know", "sq79dc_heathrow or london city airport (but not gatwick)", "nq135a_news_gun/knife crime"]
    else:  # confidence
        query = """SELECT borough, month, AVG(proportion) AS proportion
        FROM PAS_Borough
        where measure == '"good job" local'
        GROUP BY borough, month"""

        t_or_c = "confidence"
        q = ["q15_very worried", "q39a_2_major problem", "q60_very poor", "q61_poor", "q62a_strongly disagree", "q62f_tend to disagree", "q79d_2", "q79g_1 not at all well", "q79i_don't know", "q79j_don't know", "sq79b_strongly agree", "sq79dc_heathrow or london city airport (but not gatwick)", "a120_tend to agree", "q132nn_social media (e.g. facebook/twitter/blogs)"]

    df_borough = pd.read_sql_query(query, conn)

    query = """SELECT * FROM PAS_questions_cleaned"""
    df = pd.read_sql_query(query, conn_clean)
    df["month"] = df.apply(lambda row: month_to_quartile(row["month"]), axis=1)
    df = df.groupby(["borough", "month"]).sum()

    query = """SELECT * FROM PAS_questions_age_q61_cleaned"""
    df_age = pd.read_sql_query(query, conn_clean)
    df_age["month"] = df_age.apply(lambda row: month_to_quartile(row["month"]), axis=1)
    df_age = df_age.groupby(["q136r", "borough", "month"]).sum()
    df_age["q61"] = df_age["q61"] / df_age["total"]
    df_age = df_age.reset_index()
    df_age = pd.merge(df_age, df_borough, on=["borough", "month"], how="inner")

    query = """SELECT * FROM PAS_questions_race_q61_cleaned"""
    df_race = pd.read_sql_query(query, conn_clean)
    df_race["month"] = df_race.apply(lambda row: month_to_quartile(row["month"]), axis=1)
    df_race = df_race.groupby(["nq147r", "borough", "month"]).sum()
    df_race["q61"] = df_race["q61"] / df_race["total"]
    df_race = df_race.reset_index()
    df_race = pd.merge(df_race, df_borough, on=["borough", "month"], how="inner")

    query = """SELECT DISTINCT borough FROM PAS_Borough"""
    boroughs = pd.read_sql_query(query, conn)["borough"].to_list()

    df = pd.merge(df, df_borough, on=["borough", "month"], how="inner")

    column_sums = df[df.columns[2:]].sum()
    columns_to_drop = column_sums[column_sums < 1000].index

    if "proportion" in columns_to_drop:
        columns_to_drop = columns_to_drop.difference(["proportion"])
    df = df.drop(columns=columns_to_drop)

    for borough in boroughs:
        print(borough)
        df_b = df[df["borough"] == borough].copy()
        df_age_b = df_age[df_age["borough"] == borough].copy()
        df_race_b = df_race[df_race["borough"] == borough].copy()
        # make_best_linear_regressions(df_b, df_age_b, df_race_b, df_b.columns[2:], borough)
        make_best_linear_regressions(df_b, df_age_b, df_race_b, q, borough, t_or_c)

    conn.close()
    conn_clean.close()


if __name__ == "__main__":
    # trust
    full_process_lin_reg_boroughs(True)

    # confidence
    full_process_lin_reg_boroughs(False)

    # conn = sqlite3.connect("data/police_data.db")
    # conn_clean = sqlite3.connect("data/cleaned_police_data.db")
    #
    # query = """SELECT borough, month, AVG(proportion) AS proportion
    # FROM PAS_Borough
    # GROUP BY borough, month"""
    # df_borough = pd.read_sql_query(query, conn)
    #
    # query = """SELECT * FROM PAS_questions_cleaned"""
    # df = pd.read_sql_query(query, conn_clean)
    # df["month"] = df.apply(lambda row: month_to_quartile(row["month"]), axis=1)
    # df = df.groupby(["borough", "month"]).sum()
    #
    # query = """SELECT * FROM PAS_questions_age_q61_cleaned"""
    # df_age = pd.read_sql_query(query, conn_clean)
    # df_age["month"] = df_age.apply(lambda row: month_to_quartile(row["month"]), axis=1)
    # df_age = df_age.groupby(["q136r", "borough", "month"]).sum()
    # df_age["q61"] = df_age["q61"] / df_age["total"]
    # df_age = df_age.reset_index()
    # df_age = pd.merge(df_age, df_borough, on=["borough", "month"], how="inner")
    # # df_age = df_age.reset_index()
    # # df_age['total respondents'] = df_age.iloc[:, 3:10].sum(axis=1)
    #
    # query = """SELECT * FROM PAS_questions_race_q61_cleaned"""
    # df_race = pd.read_sql_query(query, conn_clean)
    # df_race["month"] = df_race.apply(lambda row: month_to_quartile(row["month"]), axis=1)
    # df_race = df_race.groupby(["nq147r", "borough", "month"]).sum()
    # df_race["q61"] = df_race["q61"] / df_race["total"]
    # df_race = df_race.reset_index()
    # df_race = pd.merge(df_race, df_borough, on=["borough", "month"], how="inner")
    # # df_race = df_race.reset_index()
    # # df_race['total respondents'] = df_race.iloc[:, 3:10].sum(axis=1)
    #
    # query = """SELECT DISTINCT borough FROM PAS_Borough"""
    # boroughs = pd.read_sql_query(query, conn)["borough"].to_list()
    #
    # df = pd.merge(df, df_borough, on=["borough", "month"], how="inner")
    # # print(df)
    # # df['total respondents'] = df.iloc[:, 2:10].sum(axis=1)
    #
    # column_sums = df[df.columns[2:]].sum()
    # columns_to_drop = column_sums[column_sums < 1000].index
    #
    # if "proportion" in columns_to_drop:
    #     columns_to_drop = columns_to_drop.difference(["proportion"])
    # df = df.drop(columns=columns_to_drop)
    #
    # for borough in boroughs:
    #     print(borough)
    #     df_b = df[df["borough"] == borough].copy()
    #     df_age_b = df_age[df_age["borough"] == borough].copy()
    #     df_race_b = df_race[df_race["borough"] == borough].copy()
    #     make_best_linear_regressions(df_b, df_age_b, df_race_b, df_b.columns[2:], borough)
    #
    #
    #
    #
    #
    # conn.close()
    # conn_clean.close()
