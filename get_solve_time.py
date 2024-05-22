import pandas as pd
import sqlite3

conn = sqlite3.connect("data/police_data.db")
cursor = conn.cursor()

# query = """SELECT s.crime_id, s.month as begin_month, o.month as end_month, SUBSTRING(s.lsoa_name, 1, length(s.lsoa_name) - 5) as borough
# FROM street s, outcomes o
# WHERE s.crime_id == o.crime_id"""
#
# df = pd.read_sql_query(query, conn)
# print(df)
#
# print("transforming dates...")
# df["solved"] = df["end_month"].str[:4].astype(int) * 12 + df["end_month"].str[5:7].astype(int)
# df_sorted = df.sort_values(by="solved", ascending=False).drop_duplicates("crime_id")
# df_sorted = df_sorted.drop(columns=["solved"])
#
# df_sorted.to_sql("solve_time_c", conn, if_exists="replace", index=False)

def calc_time(begin_data, end_data, current_date):
    # convert dates into numbers
    begin_count = int(begin_data[:4]) * 12 + int(begin_data[5:7])
    end_count = int(end_data[:4]) * 12 + int(end_data[5:7])
    current_count = int(current_date[:4]) * 12 + int(current_date[5:7])

    if current_count > end_count or current_count < begin_count:
        return None
    else:
        return current_count - begin_count

c = calc_time("2014-02", "2015-08", "2015-08")
print(c)


def get_solve_time(conn):
    # collect boroughs
    print("collecting boroughs...")
    query = """SELECT DISTINCT borough
    FROM PAS_borough"""
    boroughs = pd.read_sql_query(query, conn)["borough"].to_list()

    # collect dates
    print("collecting months...")
    query = """SELECT DISTINCT begin_month
    FROM solve_time_c
    ORDER BY begin_month"""
    months = pd.read_sql_query(query, conn)["begin_month"].to_list()

    # collect data
    print("collecting data...")
    query = """SELECT *
    FROM solve_time_c"""
    df = pd.read_sql_query(query, conn)

    # print("transforming dates...")
    # df["solved"] = df["end_month"].str[:4].astype(int) * 12 + df["end_month"].str[5:7].astype(int)
    # print(df[["end_month", "solved"]])
    #
    # df_sorted = df.sort_values(by='solved', ascending=False).drop_duplicates('crime_id')
    # print(df_sorted[["end_month", "solved"]])
    b_count = 0
    new_df = None

    for borough in boroughs:
        print(f"{borough}, {b_count}")
        df_b = df[df["borough"] == borough].copy()
        for month in months:
            print(month, b_count)
            if not df_b.empty:
                # df_b[f"{month}_{borough}"] = df_b.apply(lambda row: calc_time(row["begin_month"], row["end_month"], month), axis=1)
                df_b["index"] = df_b.apply(lambda row: calc_time(row["begin_month"], row["end_month"], month), axis=1)

                # print(df_b[f"{month}_{borough}"])
                # total = len(df_b)

                df_bin = pd.crosstab(df_b.index, df_b["index"])
                # print(df_bin)
                df_bin["month"] = month
                df_bin["borough"] = df_b["borough"].copy()

                df_bin = df_bin.reset_index()
                df_bin = df_bin.drop(columns=["row_0"])
                df_bin["total"] = 1
                df_bin = df_bin.groupby(["borough", "month"]).sum()
                df_bin = df_bin.reset_index()
                # print(df_bin)
                df_b = df_b.drop(columns=["index"])

                if new_df is None:
                    # print("HI")
                    new_df = df_bin.copy()
                else:
                    new_df = pd.concat([new_df, df_bin])
                # print(new_df)

        b_count += 1

    conn.close()

    conn = sqlite3.connect("data/cleaned_police_data.db")
    cursor = conn.cursor()
    new_df.to_sql("solve_time", conn, if_exists="replace", index=False)
    conn.close()

    print(new_df)


get_solve_time(conn)
