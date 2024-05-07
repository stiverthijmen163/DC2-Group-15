import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
import pickle
import os


# Function to parse the KML file and extract borough polygons
def extract_borough_polygons_from_kml(kml_str):
    # Parse the KML XML string
    root = ET.fromstring(kml_str)

    # Namespace for KML
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Iterate over all placemarks to extract polygons and borough names
    borough_polygons = []
    for placemark in root.findall('.//kml:Placemark', namespaces=ns):
        # Get the borough name from ExtendedData
        borough_name = placemark.find('.//kml:Data[@name="name"]/kml:value', namespaces=ns).text

        # Extract the coordinates
        coordinates_text = placemark.find('.//kml:coordinates', namespaces=ns).text

        # Create a list of (longitude, latitude) tuples
        coordinates = [
            tuple(map(float, coord.split(',')))[:2]  # Keep only longitude and latitude
            for coord in coordinates_text.strip().split()
        ]

        # Create and store polygon from these coordinates
        polygon = Polygon(coordinates)
        borough_polygons.append((borough_name, polygon))

    # Save the polygons
    data_folder = os.path.join(os.path.dirname(__file__), "Data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    save_path = os.path.join(data_folder, "borough_polygons.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(borough_polygons, f)


# Function to determine which borough a point belongs to
def find_borough(lat, lon):
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
