import pandas as pd
import sqlite3
from load_data_to_SQL import lower_case_data


def clean_data(table_name: str, query, columns: list) -> None:
    """
    Converts dataframes into dataframes containing the number of occurrences.
    :param table_name: name of the table to access in SQL database\
    :param query: SQL query to execute
    :param columns: list of column names to convert
    """
    print("collecting data...")
    df0 = get_borough(query)

    df = None
    total = True

    for column in columns:
        # convert every column in columns to binary
        print(f"creating binary columns on column {column}...")
        df1 = pd.crosstab(df0.index, df0[column])#, margins=True)

        # grouping the dataset
        print("setting right columns...")
        df1["month"] = df0["month"].copy()

        try:
            df1["borough"] = df0["borough"].copy()
        except:
            pass

        # drop unnecessary column
        df1 = df1.reset_index()
        df1 = df1.drop(columns=["row_0"])

        # add total row
        if total:
            df1["total"] = 1
            total = False

        # group if possible on Borough and month
        print("grouping...")
        try:
            df1 = df1.groupby(["borough", "month"]).sum()
        except:
            df1 = df1.groupby(["month"]).sum()

        # merging datasets
        if df is None:
            df = df1
        else:
            try:
                df = pd.merge(df, df1, on=["borough", "month"])
            except:
                df = pd.merge(df, df1, on=["month"])

    # make columns to easier access
    df = lower_case_data(df)

    # convert df top database
    df.to_sql(f"{table_name}_cleaned", conn_clean, if_exists="replace", index=True)
    conn_clean.commit()
    print(f"CSV data successfully imported into SQLite database: data/cleaned_police_data.db, table: {table_name}_cleaned\n")


def get_borough(query) -> pd.DataFrame:
    """
    Executes a SQL query.
    :param query: SQL query to execute
    :return: dataframe containing result of the SQL query
    """
    df = pd.read_sql_query(query, conn)
    return df


def remove_outside_borders() -> None:
    """
    Removes all entries that do not take e place in London (the area of interest).
    """
    table_names = ["outcomes", "stop_and_search", "street"]

    for table_name in table_names:
        # collect the data containing only the entries that take place in London
        print(f"removing data out of London from: {table_name}")
        try:  # if borough columns exists
            query = f"""SELECT * FROM {table_name} WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))"""
            df = pd.read_sql_query(query, conn)
        except:  # use LSOA code otherwise
            query = f"""SELECT *, SUBSTRING(lsoa_name, 1, length(lsoa_name) - 5) AS borough
            FROM {table_name}
            WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))"""
            df = pd.read_sql_query(query, conn)
            df = df.drop(columns=["borough"])

        # lowercase all the data
        df = lower_case_data(df)

        # save dataset to the database
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def col_to_float(answer: str) -> float:
    """
    Converts a column into numerical values, specific use for q61, ranging from 0 to 1.
    :param answer: answer to the question from the column
    """
    if answer == "very poor":
        result = 0
    elif answer == "poor":
        result = 0.25
    elif answer == "fair":
        result = 0.5
    elif answer == "good":
        result = 0.75
    elif answer == "excellent":
        result = 1
    else:  # answer does not fit q61
        result = None

    return result


if __name__ == '__main__':
    # connect to the databases
    global conn
    global conn_clean
    conn = sqlite3.connect("data/police_data.db")
    conn_clean = sqlite3.connect("data/cleaned_police_data.db")

    # remove all entries not from London
    remove_outside_borders()

    # convert outcomes dataset
    table_name = "outcomes"
    query = f'''SELECT crime_id, month, SUBSTRING(lsoa_name, 1, length(lsoa_name) - 5) AS borough, outcome_type
    FROM {table_name}'''
    clean_data(table_name, query, ["outcome_type"])

    # convert stop and search dataset
    table_name = "stop_and_search"
    query = f"""SELECT type, gender, age_range, self_defined_ethnicity, officer_defined_ethnicity, legislation, object_of_search, outcome, month, borough
    FROM {table_name}
    """
    clean_data(table_name, query, ["type", "gender", "age_range", "self_defined_ethnicity", "officer_defined_ethnicity", "legislation", "object_of_search", "outcome"])

    # convert street dataset
    table_name = "street"
    query = f'''SELECT crime_id, month, SUBSTRING(lsoa_name, 1, length(lsoa_name) - 5) AS borough, crime_type
    FROM {table_name}'''
    clean_data(table_name, query, ["crime_type"])

    # -------------------------------------- PAS Survey ----------------------------------------------------------------
    # collect all data from the PAS survey and reformat it
    df_survey = pd.read_sql_query("SELECT * FROM PAS_questions", conn)
    df_survey['borough'] = df_survey['borough'].replace('kensington & chelsea', 'kensington and chelsea')
    df_survey['borough'] = df_survey['borough'].replace('barking & dagenham', 'barking and dagenham')
    df_survey['borough'] = df_survey['borough'].replace('hammersmith & fulham', 'hammersmith and fulham')
    df_survey = df_survey.replace({'-': None, 'not asked': None})
    df_survey = df_survey.drop(['financialyear', 'ward', 'ward_n', 'soa1', 'soa2'], axis=1)

    # dataset based on age ranges
    table_name = "PAS_questions_age_q61"
    df_survey_age = df_survey.copy()
    df_survey_age["q61"] = df_survey_age.apply(lambda row: col_to_float(row["q61"]), axis=1)  # convert q61 to numerical data
    df_survey_age = pd.get_dummies(df_survey_age, columns=df_survey_age.columns[2:].difference(["q136r", "q61"]))  # create a binary column for every answer
    df_survey_age["total"] = 1  # add total row
    df_survey_age = df_survey_age.groupby(["q136r", "month", "borough"]).sum()  # group by age range, month and borough
    df_survey_age = df_survey_age.reset_index()

    # convert df based on ages to SQL db
    df_survey_age.to_sql(f"{table_name}_cleaned", conn_clean, if_exists="replace", index=False)
    conn_clean.commit()
    print(f"CSV data successfully imported into SQLite database: data/cleaned_police_data.db, table: {table_name}_cleaned\n")

    # dataset base on races
    table_name = "PAS_questions_race_q61"
    df_survey_race = df_survey.copy()
    df_survey_race["q61"] = df_survey_race.apply(lambda row: col_to_float(row["q61"]), axis=1)  # convert q61 to numerical data
    df_survey_race = pd.get_dummies(df_survey_race, columns=df_survey_race.columns[2:].difference(["nq147r", "q61"]))  # create a binary column for every answer
    df_survey_race["total"] = 1  # add total row
    df_survey_race = df_survey_race.groupby(["nq147r", "month", "borough"]).sum()  # group by age range, month and borough
    df_survey_race = df_survey_race.reset_index()

    # convert df based on races to SQL db
    df_survey_race.to_sql(f"{table_name}_cleaned", conn_clean, if_exists="replace", index=False)
    conn_clean.commit()
    print(f"CSV data successfully imported into SQLite database: data/cleaned_police_data.db, table: {table_name}_cleaned\n")

    # all data
    table_name = "PAS_questions"
    df_survey = pd.get_dummies(df_survey, columns=df_survey.columns[2:])  # create a binary column for every answer
    df_survey["total"] = 1  # add total row
    df_survey = df_survey.groupby(['month', 'borough']).sum()  # group by month and Borough
    df_survey.reset_index()

    # convert df to SQL database
    df_survey.to_sql(f"{table_name}_cleaned", conn_clean, if_exists="replace", index=True)
    conn_clean.commit()
    print(f"CSV data successfully imported into SQLite database: data/cleaned_police_data.db, table: {table_name}_cleaned\n")

    conn_clean.close()
    conn.close()
