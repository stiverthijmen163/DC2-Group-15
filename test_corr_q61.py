import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from clean_data import col_to_float
from PAS_ethnicity_q61 import month_to_quartile

# load in q61 from the database
conn = sqlite3.connect("data/police_data.db")
q = """
SELECT month, q61 FROM PAS_questions
"""

# convert 'month' and 'q61' to correct format
df_questions = pd.read_sql_query(q, conn)
df_questions["month"] = df_questions.apply(lambda row: month_to_quartile(row["month"]), axis=1)
df_questions["q61"] = df_questions.apply(lambda row: col_to_float(row["q61"]), axis=1)
df_questions = df_questions.groupby(by=["month"]).mean().reset_index()

# load and join confidence data
query = """SELECT month, AVG(proportion) AS confidence
        FROM PAS_Borough
        where measure == '"good job" local'
        GROUP BY month"""
df = pd.read_sql_query(query, conn)
df = pd.merge(df, df_questions, on=["month"], how="inner")

# load and join trust dataset
query = """SELECT month, AVG(proportion) AS trust
        FROM PAS_Borough
        where measure == 'trust mps'
        GROUP BY month"""
df_t = pd.read_sql_query(query, conn)
df = pd.merge(df, df_t, on=["month"], how="inner")

# confidence Linear Regression
X = df[["q61"]].copy()
y = df["confidence"].copy()

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
R_squared = model.score(X, y)

# plot the data
plt.figure(figsize=(8, 12))
ax1 = plt.subplot(2, 1, 1)

plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label=f'Regression line ({R_squared:.2f})')

plt.xlabel(f'Average answer to q61')
plt.ylabel('Confidence')
plt.title(f'Linear Regression of q61 and Confidence')
plt.legend()

# trust Linear Regression
X = df[["q61"]].copy()
y = df["trust"].copy()

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
R_squared = model.score(X, y)

# plot the data
ax2 = plt.subplot(2, 1, 2)

plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label=f'Regression line ({R_squared:.2f})')

plt.xlabel(f'Average answer to q61')
plt.ylabel('Trust')
plt.title(f'Linear Regression of q61 and Trust')
plt.legend()

# save the figure
plt.savefig("artifacts/TEST_corr_trust_conf.png")
