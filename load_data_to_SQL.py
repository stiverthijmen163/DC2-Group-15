import pandas as pd
import sqlite3
import os, sys
import openpyxl  # Needed to open xlsx files <DO NOT REMOVE>


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
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # All filenames containing 'outcomes'
    table_name = "outcomes"
    get_data(data_path, "outcomes")
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # All filenames containing 'stop'
    table_name = "stop_and_search"
    get_data(data_path, "stop")
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # PAS enquete
    path = "data/PAS_T&Cdashboard_to Q3 23-24.xlsx"

    # MPS table within PAS csv file
    table_name = "PAS_MPS"
    df = pd.read_excel(path, sheet_name="MPS")
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    print(f"CSV data successfully imported into SQLite database: {SQL_path}, table: {table_name}")

    # Borough table within PAS csv file
    table_name = "PAS_Borough"
    df = pd.read_excel(path, sheet_name="Borough")

    # get the correct columns
    df = df[df.columns[:6]].copy()
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


if __name__ == "__main__":
    load_data()
