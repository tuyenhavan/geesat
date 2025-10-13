import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import geopandas as gpd
from shapely.geometry import mapping
import ee


def geedate_to_python_datetime(date_code):
    """Convert GEE datetime code into Python readable datetime.

        Example:
            date_code = 673056000000 # GEE datetime code
            out_date = geedate_to_python_datetime(date_code)

    Args:
        date_code (int): The GEE datetime code.

    Returns:
        datetime.datetime: Python datetime
    """
    if not isinstance(date_code, int):
        raise TypeError("Please provide date code with integer type.")
    # Initialize the start date since GEE started date from 1970-01-01
    start_date = datetime(1970, 1, 1, 0, 0, 0)
    # Convert time code to number of hours
    hour_number = date_code / (60000 * 60)
    # Increase dates from an initial date by number of hours
    delta = timedelta(hours=hour_number)
    end_date = start_date + delta
    return end_date


def format_extracted_data(mdict):
    """A function to format the data extracted from raster by polygons FeatureCollection.

    Args:
        mdict (dict): A dictionary contains extracted data from an image or ImageCollection.

    Returns:
        pandas.DataFrame: A dataframe of extracted data with values of interest and its geometry.
    """
    mlist = []
    coords = []
    for item in mdict["features"]:
        mlist.append(item["properties"])
        coords.append(item["geometry"]["coordinates"])
    df = pd.DataFrame(mlist)
    df["coordinates"] = coords
    return df


def daily_date_list(start_year, start_month, start_day, number_of_days=365):
    """Generate a list of daily dates.

    Args:
        year (int): A start year.
        month (int): A start month.
        day (int): A start day.
        number_of_days (int, optional): A number of days to generate. Defaults to 365.

    Returns:
        list: A list of daily datetimes.
    """
    if (
        not isinstance(start_year, int)
        or not isinstance(start_month, int)
        or not isinstance(start_day, int)
    ):
        raise ValueError(
            f"year {start_year}, month {start_month}, and day {start_day} must all be integers."
        )
    first_date = datetime(start_year, start_month, start_day)
    date_list = [first_date + timedelta(days=i) for i in range(number_of_days)]
    return date_list


def weekly_date_list(start_year, start_month, start_day, number_of_week=52):
    """generate a list of weekly dates.

    Args:
        year (int): A year
        month (int): A month
        day (int): A day
        number_of_week (int, optional): A number of weeks to generate. Defaults to 52.

    Returns:
        list: A list of weekly datetimes.
    """
    if (
        not isinstance(start_year, int)
        or not isinstance(start_month, int)
        or not isinstance(start_day, int)
    ):
        raise ValueError(
            f"year {start_year}, month {start_month}, and day {start_day} must all be integers."
        )
    first_date = datetime(start_year, start_month, start_day)
    week_list = [first_date + i * timedelta(weeks=1) for i in range(number_of_week)]
    return week_list


def monthly_date_list(start_year, start_month, start_day, number_of_month=12):
    """Generate a list of monthly dates.

    Args:
        year (int): A initial year
        month (month): A start month.
        day (int): A start day.
        number_of_month (int, optional): A number of months to generate. Defaults to 12.

    Returns:
        lsit: A list of monthly dates.
    """
    first_date = datetime(start_year, start_month, start_day)
    month_list = [first_date + relativedelta(months=i) for i in range(number_of_month)]
    return month_list


def yearly_date_list(start_year, start_month, start_day, number_of_year=10):
    """Generate a list of years.

    Args:
        year (int): A start year.
        month (int): A start month.
        day (int): A start day.
        number_of_year (int, optional): A number of years to generate. Defaults to 10.

    Returns:
        list: A list of generated years.
    """
    start_date = datetime(start_year, start_month, start_day)
    year_list = [
        start_date.replace(year=start_date.year + i) for i in range(number_of_year)
    ]
    return year_list


def gdf_to_ee(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """
    Convert a GeoPandas GeoDataFrame to an Earth Engine FeatureCollection.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame. Must have a valid geometry column.

    Returns:
        ee.FeatureCollection: Equivalent Earth Engine FeatureCollection.
    """
    ee_features = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Convert shapely geometry to GeoJSON mapping
        geojson = mapping(geom)
        ee_geom = ee.Geometry(geojson)

        # Convert row properties to a dictionary (excluding geometry)
        properties = row.drop(labels="geometry").to_dict()

        # Create ee.Feature
        ee_feature = ee.Feature(ee_geom, properties)
        ee_features.append(ee_feature)

    return ee.FeatureCollection(ee_features)
