import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
import pickle
import os
import sqlite3
import pandas as pd
from load_data_to_SQL import lower_case_data


def find_borough(lat: float, lon: float) -> any:
    """
    Finds the corresponding borough to the given coordinates.
    :param lat: latitude
    :param lon: longitude
    """
    # Load the polygons
    data_folder = os.path.join(os.path.dirname(__file__), "Data")
    load_path = os.path.join(data_folder, "borough_polygons.pkl")
    with open(load_path, 'rb') as f:
        borough_polygons = pickle.load(f)

    # Create a Point object from the given latitude and longitude
    point = Point(lon, lat)

    # Iterate over all borough polygons to check which one contains the point
    for borough_name, polygon in borough_polygons:
        if polygon.contains(point):
            return borough_name

    return None


if __name__ == "__main__":
    # connect to database
    conn = sqlite3.connect("data/police_data.db")

    # collect the data to convert the coordinates from
    query = """
    SELECT *
    FROM stop_and_search
    WHERE longitude not NULL AND latitude not NULL
    """
    df = pd.read_sql_query(query, conn)

    rows = df.to_dict("records")
    boroughs = []
    length = len(rows)
    count = 0

    # find borough corresponding to each row
    for row in rows:
        borough = find_borough(row["latitude"], row["longitude"])
        boroughs.append(borough)

        # print progress
        if count % 10000 == 0:
            print(f"{100 * count /length:.3f}%")
        count += 1

    # append boroughs into dataset and lowercase the data
    df["borough"] = boroughs
    df = lower_case_data(df)

    # save dataset to database
    df.to_sql("stop_and_search", conn, if_exists="replace", index=False)
