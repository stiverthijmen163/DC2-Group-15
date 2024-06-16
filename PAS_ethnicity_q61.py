import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


def make_best_linear_regressions(df: pd.DataFrame, df_a: pd.DataFrame, df_r: pd.DataFrame, questions: list, borough: str, t_or_c: str) -> None:
    """
    Plots the questions with R^2 > 0.7 of the cleaned survey data set against the trust or confidence.
    :param df: cleaned survey data set containing all data
    :param df_a: cleaned survey data set containing all data grouped by age
    :param df_r: cleaned survey data set containing all data grouped by race
    :param questions: list of questions to plot
    :param borough: Borough name
    :param t_or_c: whether the plots contain information about trust or confidence level
    """
    # set the order of colors to use
    colors = ["blue", "orange", "green", "red", "purple", "brown"]

    # get only those answers that are in all dataframes
    combined_columns = list(set(df.columns.to_list()) & set(df_a.columns.to_list()) & set(df_r.columns.to_list()))

    ages = df_a["q136r"].unique().tolist()
    races = df_r["nq147r"].unique().tolist()

    # check if question should be considered
    for question in questions:
        if question in combined_columns:
            # set conditional statements whether a plot is worth showing (R^2 > 0.7)
            show = False
            show_race = False
            show_age = False

            # -------------------------------------- All Data ----------------------------------------------------------
            # calculate proportions
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

            # create figure containing 3 plots, access first plot
            plt.figure(figsize=(12, 18))
            ax1 = plt.subplot(3, 1, 1)

            # Plot if R-squared is in range
            if R_squared > 0.70 and len(df0) > 3:
                show = True
                plt.scatter(X, y, color='blue', label='Original data')
                plt.plot(X, y_pred, color='red', label='Regression line')
            else:  # plot data but make it barely visible
                plt.scatter(X, y, color='blue', label='Original data', alpha=0.05)
                plt.plot(X, y_pred, label=f'Regression line {R_squared:.2f}, {len(df0)}', color="red", alpha=0.05)

            # update plot layout
            plt.xlabel(f'Proportion of persons that answered {question}')
            plt.ylabel(t_or_c.capitalize())
            plt.title(f'Linear Regression of {t_or_c} and question {question}')
            plt.legend()

            # -------------------------------------------- Age ---------------------------------------------------------
            # set counter for number of data points
            age_count = 0

            # access second plot
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            for age in ages:
                # collect correct data
                df_an = df_a[df_a["q136r"] == age].copy()

                # calculate proportions
                df_an['proportion_yes'] = df_an[question] / df_an['total']
                df_an = df_an[df_an['proportion_yes'] > 0].copy()

                X = df_an[['proportion_yes']]
                y = df_an['q61']

                # Create and fit the model
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                except:
                    continue

                # Create predictions and get R^2
                y_pred = model.predict(X)
                R_squared = model.score(X, y)

                # Plot if R-squared is in range and there are at least 4 data points
                if len(df_an) > 3 and R_squared > 0.7:
                    plt.scatter(X, y, label='Original data', color=colors[ages.index(age)])
                    plt.plot(X, y_pred, label=f'Regression line {age}, {R_squared:.2f}, {len(df_an)}', color=colors[ages.index(age)])
                    age_count += 1
                else:  # plot data but make it barely visible
                    plt.plot(X, y_pred, label=f'Regression line {age}, {R_squared:.2f}, {len(df_an)}', color=colors[ages.index(age)], alpha=0.05)

                # only show plot if there are at least 2 well-fitted plots
                if age_count > 1:
                    show_age = True

                # update plot layout
                plt.xlabel(f'Proportion of persons that answered {question}')
                plt.ylabel('Average answer to q61 (0 - very poor up to 1 - excellent)')
                plt.title(f'Linear Regression of q61 and question {question}')
                plt.legend()

            # ------------------------------------------- Race ---------------------------------------------------------
            # set counter for the number of data points
            race_count = 0

            # access third plot
            ax3 = plt.subplot(3, 1, 3, sharex=ax1, sharey=ax2)

            for race in races:
                # collect correct data
                df_rn = df_r[df_r["nq147r"] == race].copy()

                # calculate proportions
                df_rn['proportion_yes'] = df_rn[question] / df_rn['total']
                df_rn = df_rn[df_rn['proportion_yes'] > 0].copy()

                X = df_rn[['proportion_yes']]
                y = df_rn['q61']

                # Create and fit the model
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                except:
                    continue

                # Create predictions and get R^2
                y_pred = model.predict(X)
                R_squared = model.score(X, y)

                # Plot if R-squared is in range and there are at least 4 data points
                if len(df_rn) > 3 and R_squared > 0.7:
                    plt.scatter(X, y, label='Original data', color=colors[races.index(race)])
                    plt.plot(X, y_pred, label=f'Regression line {race}, {R_squared:.2f}, {len(df_rn)}', color=colors[races.index(race)])
                    race_count += 1
                else:  # plot data but make it barely visible
                    plt.plot(X, y_pred, label=f'Regression line {race}, {R_squared:.2f}, {len(df_rn)}', color=colors[races.index(race)], alpha=0.05)

                # only show plot if there are at least 2 well-fitted plots
                if race_count > 1:
                    show_race = True

                # update plot layout
                plt.xlabel(f'Proportion of persons that answered {question}')
                plt.ylabel('Average answer to q61 (0 - very poor up to 1 - excellent)')
                plt.title(f'Linear Regression of q61 and question {question}')
                plt.legend()

            # save figure if at least one of the plots if worth showing
            if show or show_race or show_age:
                # capitalize borough
                plt.suptitle(f"{t_or_c} in '{' '.join([(word.capitalize() if word.lower() != 'and' else word.lower()) for word in borough.split()])}' for question {question}\n", weight='bold')

                # convert question to fit path
                question = question.replace('/', ' or ')

                plt.tight_layout()
                plt.savefig(f"artifacts/{t_or_c}/{borough}_{question}.png")
            plt.close("all")


def month_to_quartile(month: str) -> str:
    """
    Convert a date in format 'YYYY-MM' to quartiles formatted as 'YYYY/MM'.
    :param month: date in format 'YYYY-MM' to convert to quartile
    :return: quartiles formatted as 'YYYY-MM'
    """
    # extract month to convert
    month_c = int(month[5:7])

    if month_c <= 3:  # first quartile
        q = "03"
    elif month_c <= 6:  # second quartile
        q = "06"
    elif month_c <= 9:  # third quartile
        q = "09"
    else:  # fourth quartile
        q = "12"

    return f"{month[:4]}-{q}"


def full_process_lin_reg_boroughs(trust_or_conf: bool) -> None:
    """
    Collects all data based for trust or confidence and plot regression models based on age and race, via q61.
    :param trust_or_conf: whether to use the trust or confidence level
    """
    # connect to databases
    conn = sqlite3.connect("data/police_data.db")
    conn_clean = sqlite3.connect("data/cleaned_police_data.db")

    # set SQL-query to collect trust/confidence levels and relevant question based on trust or confidence
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

    # collect data
    df_borough = pd.read_sql_query(query, conn)

    # collect all cleaned data regarding PAS questions
    query = """SELECT * FROM PAS_questions_cleaned"""
    df = pd.read_sql_query(query, conn_clean)
    df["month"] = df.apply(lambda row: month_to_quartile(row["month"]), axis=1)  # convert months to quartiles
    df = df.groupby(["borough", "month"]).sum()  # group by borough and month
    df = pd.merge(df, df_borough, on=["borough", "month"], how="inner")  # combine data with trust/confidence levels

    # collect all cleaned data based on age ranges and q61
    query = """SELECT * FROM PAS_questions_age_q61_cleaned"""
    df_age = pd.read_sql_query(query, conn_clean)
    df_age["month"] = df_age.apply(lambda row: month_to_quartile(row["month"]), axis=1)  # convert months to quartiles
    df_age = df_age.groupby(["q136r", "borough", "month"]).sum()  # group by age, borough and month
    df_age["q61"] = df_age["q61"] / df_age["total"]  # calculate average answer to q61
    df_age = df_age.reset_index()
    df_age = pd.merge(df_age, df_borough, on=["borough", "month"], how="inner")  # combine data with trust/confidence levels

    # collect all cleaned data based on races and q61
    query = """SELECT * FROM PAS_questions_race_q61_cleaned"""
    df_race = pd.read_sql_query(query, conn_clean)
    df_race["month"] = df_race.apply(lambda row: month_to_quartile(row["month"]), axis=1)  # convert months to quartiles
    df_race = df_race.groupby(["nq147r", "borough", "month"]).sum()  # group by race, borough and month
    df_race["q61"] = df_race["q61"] / df_race["total"]  # calculate average answer to q61
    df_race = df_race.reset_index()
    df_race = pd.merge(df_race, df_borough, on=["borough", "month"], how="inner")  # combine data with trust/confidence levels

    # collect all boroughs
    query = """SELECT DISTINCT borough FROM PAS_Borough"""
    boroughs = pd.read_sql_query(query, conn)["borough"].to_list()

    # remove all questions answered less than 1000 times
    column_sums = df[df.columns[2:]].sum()
    columns_to_drop = column_sums[column_sums < 1000].index

    # delete 'proportion' from questions to remove if needed
    if "proportion" in columns_to_drop:
        columns_to_drop = columns_to_drop.difference(["proportion"])
    df = df.drop(columns=columns_to_drop)

    for borough in boroughs:
        print(borough)

        # collect the data for the selected borough
        df_b = df[df["borough"] == borough].copy()
        df_age_b = df_age[df_age["borough"] == borough].copy()
        df_race_b = df_race[df_race["borough"] == borough].copy()

        # run the linear regression models
        make_best_linear_regressions(df_b, df_age_b, df_race_b, q, borough, t_or_c)

    conn.close()
    conn_clean.close()


if __name__ == "__main__":
    # trust
    full_process_lin_reg_boroughs(True)

    # confidence
    full_process_lin_reg_boroughs(False)
