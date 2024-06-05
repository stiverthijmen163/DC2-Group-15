import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from PAS_ethnicity import month_to_quartile


def col_to_bin(answer: str, true_vals: list) -> bool:
    if answer in true_vals:
        result = True
    else:
        result = False

    return result


conn = sqlite3.connect("data/police_data.db")

# query = """SELECT month, AVG(mps) AS mps
# FROM PAS_Borough
# GROUP BY month"""
# df_mps = pd.read_sql_query(query, conn)
# df_mps["month"] = pd.to_datetime(df_mps["month"])
# print(df_mps)
#
# query = """SELECT month, q61
# FROM PAS_questions"""
# df_q = pd.read_sql_query(query, conn)
# df_q["total"] = 1
# df_q["month"] = df_q.apply(lambda row: month_to_quartile(row["month"]), axis=1)
# df_q["q61"] = df_q.apply(lambda row: col_to_bin(row["q61"], ["excellent", "good"]), axis=1)
# df_q["month"] = pd.to_datetime(df_q["month"])
# df_q = df_q.groupby(["month"]).sum().reset_index()
# df_q["proportion"] = df_q["q61"] / df_q["total"]
# print(df_q)
#
# plt.plot(df_mps['month'], df_mps['mps'], label='MPS')
# plt.plot(df_q['month'], df_q['proportion'], label='calculated')
# plt.legend()
# plt.show()


