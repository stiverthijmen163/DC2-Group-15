import pandas as pd
import sqlite3
import os, sys
import openpyxl  # Needed to open xlsx files <DO NOT REMOVE>
import numpy as np


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
    df = df.drop(columns=["date"])

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

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
            df0 = pd.read_csv(f"{path}/{file}")

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
    return df


if __name__ == "__main__":
    load_data()
