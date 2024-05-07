import pandas as pd
import sqlite3
from load_data_to_SQL import lower_case_data


def clean_data(table_name: str, query, columns: list) -> None:
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
        df1 = df1.reset_index()
        df1 = df1.drop(columns=["row_0"])
        if total:
            df1["total"] = 1
            total = False

        print("grouping...")
        try:
            df1 = df1.groupby(["borough", "month"]).sum()
        except:
            df1 = df1.groupby(["month"]).sum()

        # merging dataset
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
    df = pd.read_sql_query(query, conn)
    return df

def remove_outside_borders():
    table_names = ["outcomes", "stop_and_search", "street"]

    for table_name in table_names:
        print(f"removing data out of London from: {table_name}")
        try:
            query = f"""SELECT * FROM {table_name} WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))"""
            df = pd.read_sql_query(query, conn)
        except:
            query = f"""SELECT *, SUBSTRING(lsoa_name, 1, length(lsoa_name) - 5) AS borough
            FROM {table_name}
            WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))"""
            df = pd.read_sql_query(query, conn)
            df = df.drop(columns=["borough"])

        df = lower_case_data(df)

        df.to_sql(table_name, conn, if_exists="replace", index=False)


if __name__ == '__main__':
    global conn
    global conn_clean
    conn = sqlite3.connect("data/police_data.db")
    conn_clean = sqlite3.connect("data/cleaned_police_data.db")

    remove_outside_borders()

    table_name = "outcomes"
    query = f'''SELECT crime_id, month, SUBSTRING(lsoa_name, 1, length(lsoa_name) - 5) AS borough, outcome_type
    FROM {table_name}
    WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))'''
    clean_data(table_name, query, ["outcome_type"])

    table_name = "stop_and_search"
    query = f"""SELECT type, gender, age_range, self_defined_ethnicity, officer_defined_ethnicity, legislation, object_of_search, outcome, month, borough
    FROM {table_name}
--     WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))
    """
    clean_data(table_name, query, ["type", "gender", "age_range", "self_defined_ethnicity", "officer_defined_ethnicity", "legislation", "object_of_search", "outcome"])

    table_name = "street"
    query = f'''SELECT crime_id, month, SUBSTRING(lsoa_name, 1, length(lsoa_name) - 5) AS borough, crime_type
    FROM {table_name}
    WHERE borough IN ((SELECT DISTINCT borough FROM PAS_Borough))'''
    clean_data(table_name, query, ["crime_type"])

    conn_clean.close()
    conn.close()
