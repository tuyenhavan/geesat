import ee

from geesat import geogee


def prepare_flux_input_data(roi, start_date="2020-05-01", end_date="2020-10-31"):
    """Prepares input data for flux estimation by combining Sentinel-2 and Sentinel-1 data.
    Args:
        roi (ee.Geometry): Region of interest.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        ee.Image: Combined image with selected bands from Sentinel-2 and Sentinel-1.
    """
    sen2data = geogee.sen2_cloud_mask(
        aoi=roi, start_date=start_date, end_date=end_date, cloud_filter=70
    )
    sen2data = geogee.scale_data(sen2data, scale_factor=0.0001)
    sen2data = sen2data.map(
        lambda img: img.addBands(
            img.expression(
                "((B8 - B4) / (B8 + B4 + 0.0001)) * (1 + 0.5)",
                {"B8": img.select("B8"), "B4": img.select("B4")},
            ).rename("SAVI")
        )
    )
    sen2data = sen2data.map(
        lambda img: img.addBands(
            img.expression(
                "((B8 - B4) / (B8 + B4 + 0.0001)) * (1 + 0.5)",
                {"B8": img.select("B8"), "B4": img.select("B4")},
            ).rename("EVI")
        )
    )
    sen2data = sen2data.select(["B5", "B11", "SAVI", "EVI"]).median()
    sen1data = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .select(["VV", "VH"])
    ).median()
    return sen2data.addBands(sen1data).toFloat()
