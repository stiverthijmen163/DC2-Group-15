import pandas as pd
import sqlite3
import os, sys
import openpyxl  # Needed to open xlsx files <DO NOT REMOVE>
import numpy as np
import xlrd  # Needed to open xls files <DO NOT REMOVE>


def load_data():
    """"
    Loads in all police data and puts it into a SQL database.
    """
    # set paths
    SQL_path = "data/police_data.db"
    data_path = "data/mps"

    # Create a SQLite connection and cursor
    conn = sqlite3.connect(SQL_path)
    cursor = conn.cursor()

    # All filenames containing 'street'
    table_name = "street"
    df = get_data(data_path, "street")

    # clean the data
    df = lower_case_data(df)
    df["year"] = df["month"].astype(str).str[:4]
    df = df[df["year"].astype(int) >= 2014]
    df = df.drop(columns=["year"])
    df = df.drop_duplicates(subset=["crime_id"], keep="last")

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # All filenames containing 'outcomes'
    table_name = "outcomes"
    df = get_data(data_path, "outcomes")

    # clean the data
    df = lower_case_data(df)
    df["year"] = df["month"].astype(str).str[:4]
    df = df[df["year"].astype(int) >= 2014]
    df = df.drop(columns=["year"])
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # All filenames containing 'stop'
    table_name = "stop_and_search"
    df = get_data(data_path, "stop")

    # clean the data
    df = lower_case_data(df)
    df["month"] = df["date"].astype(str).str[:7]
    df = df.drop(columns=["date"])

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # PAS enquete
    path = "data/PAS_T&Cdashboard_to Q3 23-24.xlsx"

    # MPS table within PAS csv file
    table_name = "PAS_MPS"
    df = pd.read_excel(path, sheet_name="MPS")

    # clean the data
    df = lower_case_data(df)
    df["month"] = df["date"].astype(str).str[:7]
    df = df.drop(columns=["date"])

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # Borough table within PAS csv file
    table_name = "PAS_Borough"
    df = pd.read_excel(path, sheet_name="Borough")

    # get the correct columns
    df = df[df.columns[:6]].copy()

    # clean the data
    df = lower_case_data(df)
    df["month"] = df["date"].astype(str).str[:7]
    df = df.replace(to_replace="city of westminster", value="westminster")
    df = df.drop(columns=["date"])

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # Questions table within MPS csv file
    table_name = "PAS_questions"
    df = get_data(data_path, "PAS_ward_level")

    # clean the data
    df.reset_index(drop=True, inplace=True)
    df = df.drop(
        ['Unnamed: 0', 'ward_unique', 'Borough2', 'BOROUGHNEIGHBOURHOOD', 'WARD_1', 'WARD_0', 'BOROU0',
         'BOROU1', 'BOROUGHNEIGHBOURHOODCODED', 'Quarter1.1',
         'Quarter', 'interview_date', 'quarter'], axis=1)  # drop redundant columns
    df = lower_case_data(df)
    df = df.dropna(axis=1, thresh=int((0.5 * len(df))))  # drop all columns that have more than 50% of NA values

    # Formatting month column
    df['month'] = df['month'].str.replace(r'.*?\((.*?)\).*', r'\1', regex=True)  # only keep the values between the parentheses
    df['month'] = pd.to_datetime(df['month'], format='%b %Y').dt.strftime('%Y-%m')  # change to correct datetime format

    # Moving the borough column to a better position
    column_to_move = df.pop('borough')
    df.insert(2, 'borough', column_to_move)

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # unemployment rate
    table_name = "employment"
    path = "data/employment-disability-borough.xls"
    new_df = None

    query = """SELECT DISTINCT borough
    FROM PAS_borough"""
    boroughs = pd.read_sql_query(query, conn)["borough"].to_list()

    for year in range(2014, 2023):
        # load data for 'year'
        df = pd.read_excel(path, sheet_name=str(year))

        # clean the data
        df = df[["Unnamed: 1", "Unnamed: 84"]]
        df = df.set_axis(["borough", "employment"], axis=1)
        df = lower_case_data(df)
        df = df.dropna()
        # df.iloc[33, 0] = "city of westminster"
        df = df[df["borough"].isin(boroughs)].copy().reset_index(drop=True)

        # concatenate dataframes
        df["year"] = year
        if new_df is None:
            new_df = df.copy()
        else:
            new_df = pd.concat([new_df, df])

    # group df
    new_df["employment"] = new_df["employment"].astype(float)
    df = new_df.groupby(["borough", "year"]).mean()

    # save df to SQL
    SQL_path = "data/cleaned_police_data.db"
    conn_clean = sqlite3.connect(SQL_path)
    df.to_sql(table_name, conn_clean, index=True, if_exists="replace")
    conn_clean.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    conn_clean.close()
    conn.close()


def get_data(path: str, word: str) -> pd.DataFrame:
    """
    Gets all files within a path, containing the word in its filename, no maps/folders are considered,
    all files must be csv files.
    :param path: path to the root folder
    :param word: string that must be contained in the filename
    :return: dataframe containing all files containing 'word' in the filename
    """
    # set directory
    dir = os.listdir(path)
    df = None

    # iterate over all files within dir
    for file in dir:
        if word in file:
            print(file)
            df0 = pd.read_csv(f"{path}/{file}", low_memory=False)

            # create one large dataframe
            if df is None:
                df = df0.copy()
            else:
                df = pd.concat([df, df0])

    return df


def lower_case_data(df: pd.DataFrame) -> pd.DataFrame:
    # columns = [i.replace(" ", "_").lower() for i in df.columns.to_list()]
    # df = df.set_axis(columns, axis=1)
    columns = [i.replace("-", " ").lower() for i in df.columns.to_list()]
    df = df.set_axis(columns, axis=1)

    columns = ['_'.join(i.lower().split()) for i in df.columns.to_list()]
    df = df.set_axis(columns, axis=1)

    for column in columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.lower()
            df[column] = df[column].replace("nan", None)
            df[column] = df[column].replace('None', None)
            df[column] = df[column].replace('none', None)
    return df


if __name__ == "__main__":
    load_data()
