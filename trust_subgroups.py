import sqlite3
import pandas as pd


def group_ethnicity(row):
    ethnicity = row['nq147r']
    if pd.isnull(ethnicity):
        return 'Unknown'
    ethnicity = ethnicity.lower()
    if 'white' in ethnicity:
        return 'White'
    elif 'black' in ethnicity:
        return 'Black'
    elif 'asian' in ethnicity or 'chinese' in ethnicity:
        return 'Asian'
    elif 'mixed' in ethnicity or 'other' in ethnicity or 'not stated' in ethnicity:
        return 'Mixed/Other'
    else:
        return 'Unknown'


if __name__ == "__main__":
    # Connect to db
    cnx = sqlite3.connect('data/police_data.db')
    df_survey = pd.read_sql_query("SELECT * FROM PAS_questions", cnx)


    # Apply the function to the ethnicity (nq147r) column
    df_survey['nq147r'] = df_survey.apply(group_ethnicity, axis=1)

    df_survey_with_dummies = pd.get_dummies(df_survey, columns=['q61'], prefix='q61')

    # Group by age range (q136r) and ethnicity (nq147r)
    df_grouped = df_survey_with_dummies.groupby(['q136r', 'nq147r']).sum().reset_index()
    df_grouped['total'] = df_grouped[['q61_excellent', 'q61_good', 'q61_fair', 'q61_poor', 'q61_very poor']].sum(axis=1)

    # Define the trust values for each category
    trust_values = {
        'q61_excellent': 1.00,
        'q61_good': 0.75,
        'q61_fair': 0.50,
        'q61_poor': 0.25,
        'q61_very poor': 0.00
    }

    # Calculate the average trust level
    df_grouped['average_trust'] = sum(df_grouped[col] * value for col, value in trust_values.items()
                                      ) / df_grouped['total']

    # rename columns and order by average_trust and drop unknown ethnicity
    df_grouped = df_grouped.rename(columns={'q136r': 'age_range', 'nq147r': 'ethnicity'})
    df_grouped = df_grouped.sort_values(by='average_trust', ascending=True)
    df_grouped = df_grouped[df_grouped['ethnicity'] != 'Unknown']
    print(df_grouped)

    cnx.close()
