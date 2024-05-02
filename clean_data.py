import pandas as pd
import sqlite3


def clean_data(table_name: str) -> None:
    print("collecting data...")
    df0 = get_borough(table_name)

    print("creating binary columns...")
    df1 = pd.crosstab(df0.index, df0.crime_type)

    print("setting right columns...")
    df1["Month"] = df0["Month"].copy()
    df1["Borough"] = df0["Borough"].copy()
    df1 = df1.reset_index()
    df1 = df1.drop(columns=["row_0"])

    print("grouping...")
    df1 = df1.groupby(["Month", "Borough"]).sum()

    df1.to_sql(f"{table_name}_cleaned", conn_clean, if_exists="replace", index=True)
    conn_clean.commit()
    print(f"CSV data successfully imported into SQLite database: data/cleaned_police_data.db, table: {table_name}_cleaned")


def get_borough(table_name: str) -> pd.DataFrame:
    query = f'''SELECT "Crime ID" AS crime_ID, Month, SUBSTRING("LSOA name", 1, length("LSOA name") - 5) AS Borough, "Crime Type" AS crime_type
    FROM {table_name}
    WHERE Borough IN ((SELECT DISTINCT Borough FROM PAS_Borough))'''

    df = pd.read_sql_query(query, conn)
    return df


if __name__ == '__main__':
    global conn
    global conn_clean
    conn = sqlite3.connect("data/police_data.db")
    conn_clean = sqlite3.connect("data/cleaned_police_data.db")

    # clean_data("outcomes")
    # clean_data("stop_and_search")
    clean_data("street")

    conn_clean.close()
    conn.close()
