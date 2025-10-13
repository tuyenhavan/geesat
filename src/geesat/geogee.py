import ee
from geesat import common
import pandas as pd


################################## Vegetation Indices ##################################
def calculate_ndvi(image, nir_band="B8", red_band="B4"):
    """Calculate NDVI from an image or an ImageCollection.

    Args:
        image (ee.Image|ee.ImageCollection): The input image or ImageCollection.
        nir_band (str, optional): The name of the NIR band. Defaults to "B8".
        red_band (str, optional): The name of the red band. Defaults to "B4".
    """
    if isinstance(image, ee.ImageCollection):
        ndvi = image.map(
            lambda img: img.expression(
                "(NIR - RED) / (NIR + RED)",
                {
                    "NIR": img.select(nir_band),
                    "RED": img.select(red_band),
                },
            ).rename("NDVI")
        )
    elif isinstance(image, ee.Image):
        ndvi = image.expression(
            "(NIR - RED) / (NIR + RED)",
            {
                "NIR": image.select(nir_band),
                "RED": image.select(red_band),
            },
        ).rename("NDVI")
    else:
        raise TypeError(
            "Unsupported data type. It only supports ee.Image or ee.ImageCollection"
        )
    return ndvi


def calculate_evi(image, nir_band="B8", red_band="B4", blue_band="B2"):
    """Calculate EVI from an image or an ImageCollection.

    Args:
        image (ee.Image|ee.ImageCollection): The input image or ImageCollection.
        nir_band (str, optional): The name of the NIR band. Defaults to "B8".
        red_band (str, optional): The name of the red band. Defaults to "B4".
        blue_band (str, optional): The name of the blue band. Defaults to "B2".

    Returns:
        ee.Image|ee.ImageCollection: The NDVI image or ImageCollection.
    """
    if isinstance(image, ee.ImageCollection):
        evi = image.map(
            lambda img: img.expression(
                "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
                {
                    "NIR": img.select(nir_band),
                    "RED": img.select(red_band),
                    "BLUE": img.select(blue_band),
                },
            ).rename("EVI")
        )
    elif isinstance(image, ee.Image):
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select(nir_band),
                "RED": image.select(red_band),
                "BLUE": image.select(blue_band),
            },
        ).rename("EVI")
    else:
        raise TypeError(
            "Unsupported data type. It only supports ee.Image or ee.ImageCollection"
        )
    return evi


def calculate_ndwi(image, nir_band="B8", swir_band="B11"):
    """Calculate NDWI from an image or an ImageCollection.

    Args:
        image (ee.Image|ee.ImageCollection): The input image or ImageCollection.
        nir_band (str, optional): The name of the NIR band. Defaults to "B8".
        swir_band (str, optional): The name of the SWIR band. Defaults to "B11".

    Returns:
        ee.Image|ee.ImageCollection: The NDWI image or ImageCollection.
    """
    if isinstance(image, ee.ImageCollection):
        ndwi = image.map(
            lambda img: img.expression(
                "(NIR - SWIR) / (NIR + SWIR)",
                {
                    "NIR": img.select(nir_band),
                    "SWIR": img.select(swir_band),
                },
            ).rename("NDWI")
        )
    elif isinstance(image, ee.Image):
        ndwi = image.expression(
            "(NIR - SWIR) / (NIR + SWIR)",
            {
                "NIR": image.select(nir_band),
                "SWIR": image.select(swir_band),
            },
        ).rename("NDWI")
    else:
        raise TypeError(
            "Unsupported data type. It only supports ee.Image or ee.ImageCollection"
        )
    return ndwi


def calculate_savi(image, nir_band="B8", red_band="B4", L=0.5):
    """Calculate SAVI from an image or an ImageCollection.

    Args:
        image (ee.Image|ee.ImageCollection): The input image or ImageCollection.
        nir_band (str, optional): The name of the NIR band. Defaults to "B8".
        red_band (str, optional): The name of the red band. Defaults to "B4".
        L (float, optional): The soil brightness correction factor. Defaults to 0.5.

    Returns:
        ee.Image|ee.ImageCollection: The SAVI image or ImageCollection.
    """
    if isinstance(image, ee.ImageCollection):
        savi = image.map(
            lambda img: img.expression(
                "(NIR - RED) / (NIR + RED + L) * (1 + L)",
                {
                    "NIR": img.select(nir_band),
                    "RED": img.select(red_band),
                    "L": L,
                },
            ).rename("SAVI")
        )
    elif isinstance(image, ee.Image):
        savi = image.expression(
            "(NIR - RED) / (NIR + RED + L) * (1 + L)",
            {
                "NIR": image.select(nir_band),
                "RED": image.select(red_band),
                "L": L,
            },
        ).rename("SAVI")
    else:
        raise TypeError(
            "Unsupported data type. It only supports ee.Image or ee.ImageCollection"
        )
    return savi


def calculate_sen1_indices(col):
    """Calculate Sentinel-1 indices for an ImageCollection.
    Args:
        col (ee.ImageCollection): Sentinel-1 ImageCollection.
    Returns:
        ee.ImageCollection: ImageCollection with added indices.
    """

    def add_indices(img):
        img = img.select(["VV", "VH"])
        # 4 × VH / (VV + VH)
        rvi = img.expression(
            "4 * VH / (VV + VH)", {"VV": img.select("VV"), "VH": img.select("VH")}
        ).rename("RVI")
        # vh/vv ratio
        vh_vv_ratio = img.select("VH").divide(img.select("VV")).rename("VH_VV_Ratio")
        return img.addBands([rvi, vh_vv_ratio])

    return col.map(add_indices)


def calculate_landsat_indices(
    filter_bound, start_date="2020-10-01", end_date="2023-12-31"
):
    """Calculate Landsat indices for a given date range and bounding area.
    Args:
        filter_bound (ee.Geometry): The bounding area to filter the Landsat collection.
        start_date (str): Start date for filtering the Landsat collection in 'YYYY-MM-DD' format.
        end_date (str): End date for filtering the Landsat collection in 'YYYY-MM-DD' format.
    Returns:
        ee.ImageCollection: Landsat collection with added indices.
    """
    import geopandas as gpd

    if isinstance(filter_bound, gpd.GeoDataFrame):
        filter_bound = common.gdf_to_ee(filter_bound)
    ls9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(filter_bound)
        .filterDate(start_date, end_date)
    )
    ls8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(filter_bound)
        .filterDate(start_date, end_date)
    )

    landsat = ls8.merge(ls9).sort("system:time_start")

    def mask_landsat_cloud(img):
        cloud_shadow_bit_mask = 1 << 3
        cloud_bit_mask = 1 << 5
        qa = img.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cloud_bit_mask).eq(0))
        )
        return img.updateMask(mask)

    landsat = landsat.map(mask_landsat_cloud)

    def add_indices(image):
        img = (
            image.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"])
            .multiply(0.0000275)
            .add(-0.2)
        )
        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        lst = (
            image.select("ST_B10")
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename("LST")
        )
        ndwi = img.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
        # calculate TVI
        tvx = lst.divide(ndvi).rename("TVX")
        # VTI VTI = NDVI × LST
        vti = ndvi.multiply(lst).rename("VTI")
        return img.addBands([ndvi, lst, ndwi, tvx, vti]).copyProperties(
            image, ["system:time_start"]
        )

    return landsat.map(add_indices)


################################## Cloud Masking ##################################


def bitwise_extract(img, from_bit, to_bit):
    """Extract cloud-related bits

    Args:
        img (ee.Image): The input image containing QA bands.
        from_bit (int): The starting bit.
        to_bit (int): The ending bit (inclusive).

    Returns:
        ee.Image: The output image with wanted bit extracts.
    """
    mask_size = ee.Number(to_bit).add(ee.Number(1)).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    out_img = img.rightShift(from_bit).bitwiseAnd(mask)
    return out_img


def cloud_mask(col, from_bit, to_bit, qa_band_name, threshold=1):
    """Mask out cloud-related pixels from an ImageCollection.

    Args:
        col (ee.ImageCollection): The input image collection.
        from_bit (int): The starting bit to extract bitmask.
        to_bit (int): The ending bit value to extract bitmask.
        QA_band (str): The quality assurance band, which contains cloud-related information.
        threshold (int|optional): The threshold that retains cloud-free pixels.

    Returns:
        ee.ImageCollection: Cloud masked ImageCollection.
    """

    def img_mask(img):
        qa_band = img.select(qa_band_name)
        bitmask_band = bitwise_extract(qa_band, from_bit, to_bit)
        mask_threshold = bitmask_band.lte(threshold)
        masked_band = img.updateMask(mask_threshold)
        return masked_band

    cloudless_col = col.map(img_mask)
    return cloudless_col


def modis_cloud_mask(col, from_bit, to_bit, qa_band="DetailedQA", threshold=1):
    """Return a collection of MODIS cloud-free images

    Args:
        col (ee.ImageCollection): The input image collection.
        from_bit (int): The start bit to extract.
        to_bit (int): The last bit to extract.
        QA_band (str|optional): The quality band which contains cloud-related infor. Default to DetailedQA.
        threshold (int|optional): The threshold value to mask cloud-related pixels. Default to 1.

    Returns:
        ee.ImageCollection: The output collection with cloud-free pixels.
    """
    if not isinstance(col, ee.ImageCollection):
        raise TypeError("Unsupported data type. It only supports ee.ImageCollection")
    out_col = cloud_mask(col, from_bit, to_bit, qa_band, threshold)
    return out_col


def landsat_cloud_mask(collection):
    """
    Applies a cloud and cloud shadow mask to a Landsat ImageCollection using the QA_PIXEL band.
    
    The function removes pixels flagged as clouds or cloud shadows based on the QA_PIXEL band
    in Landsat imagery. This function mainly works with Landsat 8 and 9 level 2 collection 2 tier 1 data.

    Parameters:
        collection (ee.ImageCollection): The input Landsat ImageCollection.

    Returns:
        ee.ImageCollection: The masked ImageCollection with clouds and cloud shadows removed.

    Example:
        # Load a Landsat 8 ImageCollection
        landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                                .filterBounds(ee.Geometry.Point([106.85, 10.76])) \
                                .filterDate("2023-01-01", "2023-12-31")

        # Apply the cloud masking function
        masked_collection = mask_landsat_clouds(landsat_collection)
    """

    def mask_clouds(image):
        """Masks clouds and cloud shadows in a Landsat image using the QA_PIXEL band."""
        cloud_shadow_bit = 1 << 3  # Bit 3: Cloud shadow
        cloud_bit = 1 << 5  # Bit 5: Cloud

        qa = image.select("QA_PIXEL")
        mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(qa.bitwiseAnd(cloud_bit).eq(0))
        return image.updateMask(mask)

    return collection.map(mask_clouds)


def sen2_cloud_mask(
    aoi,
    start_date,
    end_date,
    cloud_filter=40,
    cloud_prob_threshold=40,
    sr_band_scale=1e4,
    nir_dark_thresh=0.15,
    cloud_projection_distance=1,
    buffer=50,
):
    """Applies a cloud and cloud shadow mask to a Sentinel-2 surface reflectance image collection
    for the given Area of Interest (AOI) and time range.

    The function utilizes Sentinel-2 surface reflectance (SR) data and cloud probability
    information from the COPERNICUS/S2_CLOUD_PROBABILITY collection. It adds cloud and cloud
    shadow masks to the image collection based on the specified thresholds and parameters.
    The final collection includes a mask that excludes cloud and cloud-shadow affected pixels.

    Parameters:
    ----------
    aoi : ee.Geometry
        The Area of Interest (AOI) defined as a geometry (e.g., point, polygon, etc.)
        for which the analysis is to be performed.

    start_date : str
        The start date of the time range for the image collection in 'YYYY-MM-DD' format.

    end_date : str
        The end date of the time range for the image collection in 'YYYY-MM-DD' format.

    cloud_filter : int, optional (default=40)
        The maximum acceptable percentage of cloudy pixels in the Sentinel-2 images.
        Images with higher cloud cover will be excluded from the analysis.

    cloud_prob_threshold : int, optional (default=40)
        The cloud probability threshold (in percentage) for cloud detection.
        Pixels with a cloud probability higher than this value will be identified as clouds.

    sr_band_scale : int, optional (default=1e4)
        Scaling factor applied to the surface reflectance bands for consistency across different
        sensors. Typically, Sentinel-2 surface reflectance values are scaled by a factor of 10,000.

    nir_dark_thresh : float, optional (default=0.15)
        Threshold for identifying dark NIR pixels, which are considered potential cloud shadow pixels.

    cloud_projection_distance : int, optional (default=1)
        The distance (in pixels) over which the cloud shadow will be projected from cloud pixels.
        This assumes the projection follows the solar azimuth angle.

    buffer : int, optional (default=50)
        The buffer distance (in meters) applied to the cloud-shadow mask to reduce small artifacts
        and improve mask continuity.

    Returns:
    -------
    ee.ImageCollection
        A cloud and cloud shadow-masked Sentinel-2 image collection, with the cloud and shadow
        pixels masked out. The final image collection includes bands for cloud probability,
        cloud shadow projection, and the final cloud-shadow mask.

    Example:
    --------
    aoi = ee.Geometry.Point([10.4806, 51.8012])
    start_date = '2020-05-01'
    end_date = '2020-12-31'
    cloud_filter = 10
    result = s2cloud_mask(aoi, start_date, end_date, cloud_filter)"""

    # Sentinel-2 surface reflectance
    s2sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter))
    )
    # Cloudless collection
    s2cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    # Join the collections
    col = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2sr,
                "secondary": s2cloudless,
                "condition": ee.Filter.equals(
                    **{"leftField": "system:index", "rightField": "system:index"}
                ),
            }
        )
    )

    def add_cloud_band(image):
        cloud = ee.Image(image.get("s2cloudless")).select("probability")
        is_cloud = cloud.gt(cloud_prob_threshold).rename("clouds")
        return image.addBands(ee.Image([cloud, is_cloud]))

    # add shadow band
    def add_shadow_band(image):
        # Identify water pixels from the SCL band.
        nonwater = image.select("SCL").neq(6)
        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        dark_pixels = (
            image.select("B8")
            .lt(nir_dark_thresh * sr_band_scale)
            .multiply(nonwater)
            .rename("dark_pixels")
        )
        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )
        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (
            image.select("clouds")
            .directionalDistanceTransform(
                shadow_azimuth, cloud_projection_distance * 10
            )
            .reproject(**{"crs": image.select(0).projection(), "scale": 100})
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )
        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename("shadows")
        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    # put them together and produce final mask
    def add_cloud_shadow_mask(image):
        # add cloud component bands
        img_cloud = add_cloud_band(image)
        # add cloud shadow component bands
        img_shadow = add_shadow_band(img_cloud)
        # Combine cloud and cloud shadow mask and set cloud and shadow as value 1 else 0
        is_cloud_shadow = (
            img_shadow.select("clouds").add(img_shadow.select("shadows")).gt(0)
        )
        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cloud_shadow = (
            is_cloud_shadow.focalMin(2)
            .focalMax(buffer * 2 / 20)
            .reproject(**{"crs": image.select(0).projection(), "scale": 20})
            .rename("cloud_mask")
        )
        return image.addBands(is_cloud_shadow)

    # apply cloud mask to each image
    def apply_cloud_shadow_mask(image):
        noncloud = image.select("cloud_mask").Not()
        return image.select("B.*").updateMask(noncloud)

    # apply cloud mask to the collection
    fcol = col.map(add_cloud_shadow_mask).map(apply_cloud_shadow_mask)
    return fcol


################################## Unit Conversion ##################################


def convert_landsat_lst_to_celsius(collection, roi=None, band="ST_10"):
    """
    Converts Landsat surface temperature (LST) from Kelvin to Celsius in an ImageCollection.

    The function applies a cloud mask using `mask_landsat_clouds`, optionally filters by 
    a region of interest (ROI), and converts the specified thermal band from Kelvin to Celsius.

    Parameters:
        collection (ee.ImageCollection): The input Landsat ImageCollection containing LST data.
        roi (ee.Geometry, optional): A region of interest to filter the collection. Defaults to None.
        band (str, optional): The thermal band name to process. Defaults to "ST_10" (Landsat 8/9).

    Returns:
        ee.ImageCollection: The processed ImageCollection with the LST band converted to Celsius.

    Example:
        # Define a region of interest (ROI)
        roi = ee.Geometry.Point([106.85, 10.76])

        # Load a Landsat 8 Collection
        landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                                .filterDate("2023-01-01", "2023-12-31")

        # Convert LST to Celsius
        lst_celsius_collection = convert_landsat_lst_to_celsius(landsat_collection, roi)

        # Display the first image
        Map.addLayer(lst_celsius_collection.first(), {"min": 20, "max": 50, "palette": ["blue", "green", "red"]}, "LST in Celsius")
    """
    if roi:
        collection = collection.filterBounds(roi)

    # Apply cloud masking
    collection = landsat_cloud_mask(collection)

    def to_celsius(image):
        """Converts the specified thermal band from Kelvin to Celsius."""
        lst_celsius = (
            image.select(band).multiply(0.00341802).subtract(273.15).rename("lst")
        )  # Rename for clarity
        return image.addBands(lst_celsius).copyProperties(image, ["system:time_start"])

    return collection.map(to_celsius)


def kelvin_to_celsius(col):
    """Convert temperature from Kelvin unit to celsius degree

    Args:
        col (ee.Image|ee.ImageCollection): The input image or collection for unit conversion.

    Returns:
        ee.Image|ee.ImageCollection: The converted output image or collection in celsius unit.
    """
    if isinstance(col, ee.Image):
        out_data = ee.Image(
            col.subtract(273.15).copyProperties(col, col.propertyNames())
        )
    elif isinstance(col, ee.ImageCollection):
        out_data = col.map(
            lambda img: img.subtract(273.15).copyProperties(img, img.propertyNames())
        )
    else:
        out_data = col
    return out_data


def scale_data(ds, scale_factor=1):
    """Scaling ImageCollection or Image by a specified factor.

    Example: Scaling tmax and tmin variable in TerraClimate by a factor of 0.1
    (Please see band specification for scaling factor)
    ds = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(["tmmn","tmmx"])
    outds = data_scale(ds, scale_factor=0.1)

    Args:
        ds (ee.Image|ee.ImageCollection): An ImageCollection or Image object
        scale_factor (int|float, optional): A scaling factor. Defaults to 1.

    Returns:
        ee.ImageCollection|ee.Image: An scaled ImageCollection or Image
    """

    if isinstance(ds, ee.ImageCollection):
        scaled_data = ds.map(
            lambda img: img.multiply(scale_factor).copyProperties(
                img, ["system:time_start"]
            )
        )
        return scaled_data
    if isinstance(ds, ee.Image):
        scaled_data = ds.multiply(scale_factor)
        return scaled_data


def resample_collection(col, resample_method=None, scale=None, crs=None):
    """Return a collection of resampled images. Resampling methods include max, min,
    bilinear, bicubic, average, mode, and median.

    Args:
        col (ee.ImageCollection|ee.Image): The input image collection.
        resample_method (str|optional): The resampling method. Default to bilinear.
        crs (str|optional): The coordinate referenced system (reprojection) EPSG code.
            Default to the crs of first band in collection. If not, crs = "EPSG:4326".
        scale (int|float|optional): The spatial resolution in meters. Default to 1k m.

    Returns:
        ee.ImageCollection: The output of resampled collection.
    """
    if resample_method is None:
        resample_method = "bilinear"
    if crs is None:
        if isinstance(col, ee.Image):
            crs = col.select(0).projection().getInfo()["crs"]
        elif isinstance(col, ee.ImageCollection):
            crs = col.first().select(0).projection().getInfo()["crs"]
        else:
            crs = "EPSG:4326"
    else:
        crs = "EPSG:4326"
    if scale is None:
        scale = 1000
    if not (
        isinstance(resample_method, str)
        and isinstance(crs, str)
        and isinstance(scale, (int, float))
    ):
        raise TypeError("Unsupported data type in crs and scale!")
    if isinstance(col, ee.Image):
        data = ee.Image(col).resample(resample_method).reproject(crs=crs, scale=scale)
    elif isinstance(col, ee.ImageCollection):
        data = col.map(
            lambda img: img.resample(resample_method).reproject(crs=crs, scale=scale)
        )
    else:
        raise TypeError("Unsupported data type!")
    return data


################################ Extract Raster Values ##################################


def extract_raster_values_by_polygon(ds, aoi, aggregate_method="mean", scale=1000):
    """Extracting raster values from an Image by ee.FeatureCollection.

    Args:
        ds (ee.Image): An image for extraction.
        aoi (ee.FeatureCollection): A FeatureCollection contains polygons.
        aggregate_method (str, optional): A method for aggregating raster values by polygon. Defaults to "mean".
        scale (int, optional): A scale for aggregation. Defaults to 1000.

    Returns:
        Dict: A dictionary contains extracted data and other properties.
    """
    if aggregate_method in ["max", "maximum"]:
        method = ee.Reducer.mean()
    elif aggregate_method in ["min", "minimum"]:
        method = ee.Reducer.min()
    elif aggregate_method in ["total", "sum"]:
        method = ee.Reducer.sum()
    elif aggregate_method in ["std"]:
        method = ee.Reducer.stdDev()
    else:
        method = ee.Reducer.mean()

    def polygon_value(feature):
        mean = ds.reduceRegion(reducer=method, geometry=feature.geometry(), scale=scale)
        return feature.set(mean)

    data = aoi.map(polygon_value).getInfo()
    return data


def extract_raster_values_from_collection_by_polygons(col, polygon, scale=1000):
    """
    Extracts mean raster values from an Earth Engine ImageCollection over a polygon FeatureCollection
    and returns the results as a pandas DataFrame.

    This function maps over each image in the ImageCollection, performs a zonal statistics operation
    using the provided polygons (FeatureCollection), and extracts the mean raster value per polygon
    per date.

    Parameters:
    ----------
    col : ee.ImageCollection
        An Earth Engine ImageCollection (e.g., ERA5-Land monthly temperature).

    polygon : ee.FeatureCollection
        A FeatureCollection of polygons to extract values over.

    scale : int, optional (default=1000)
        The spatial resolution (in meters) at which to perform the reduction.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame where each row represents a polygon-date pair with the corresponding
        mean raster value and any original polygon properties.

    Example:
    --------
    >>> import ee
    >>> import geopandas as gpd
    >>> from yourmodule import extract_raster_values_from_collection_by_polygons

    >>> ee.Initialize()
    >>> col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select("temperature_2m").filterDate("2020-01-01", "2020-12-31")
    >>> polygons = ee.FeatureCollection("users/your_username/your_polygon_asset")
    >>> df = extract_raster_values_from_collection_by_polygons(col, polygons, scale=1000)
    >>> print(df.head())
    """
    if not isinstance(col, ee.ImageCollection):
        raise TypeError("Unsupported data type. It only supports ee.ImageCollection")
    if not isinstance(polygon, ee.FeatureCollection):
        try:
            polygon = common.gdf_to_ee(polygon)
        except Exception as e:
            raise TypeError(
                "Unsupported data type. It only supports ee.FeatureCollection or geopandas dataframe"
            ) from e

    # Function to extract raster values from a collection by polygons
    def extract_values(image):
        # Extract the date from the image
        date = image.date().format("YYYY-MM-dd")
        # Reduce the image by the polygons
        stats = image.reduceRegions(
            collection=polygon, reducer=ee.Reducer.mean(), scale=scale
        ).map(lambda feature: feature.set("date", date))
        return stats

    # Map the function over the collection
    results = col.map(extract_values).flatten().getInfo()

    # Convert the results to a pandas DataFrame
    data = []
    for f in results["features"]:
        properties = f["properties"]
        data.append(properties)
    df = pd.DataFrame(data)
    return df


def extract_raster_values_by_point(
    points, raster, scale=10, stride=100, keep_date=True
):
    """Extract raster values at specified points.
    Args:
        points (gpd.GeoDataFrame): GeoDataFrame containing point geometries.
        raster (ee.ImageCollection): Earth Engine image from which to extract values.
        stride (int): Number of points to process in each batch.
    Returns:
        pd.DataFrame: DataFrame containing extracted raster values and timestamps.
    """
    from datetime import datetime

    import geopandas as gpd

    if not isinstance(points, gpd.GeoDataFrame):
        raise TypeError("points must be a GeoDataFrame")
    counts = len(points)
    dlist = []
    for i in range(0, counts, stride):
        start = i
        end = i + stride
        if end > counts:
            end = counts
        point = common.gdf_to_ee(points.iloc[start:end].reset_index(drop=True))
        extracted_data = raster.getRegion(geometry=point, scale=scale).getInfo()
        tdf = pd.DataFrame(extracted_data[1:], columns=extracted_data[0])
        dlist.append(tdf)
    df = pd.concat(dlist, ignore_index=True)
    df["time"] = [
        datetime.fromtimestamp(timestamp_ms / 1000) for timestamp_ms in df["time"]
    ]
    if keep_date:
        df["time"] = pd.to_datetime(df["time"]).dt.date
    return df


################################## Export Data ##################################


def export_img_to_googledrive(
    ds, aoi, folder_name="GEE_Data", file_name="NDVI_data", res=1000, crs="EPSG:4326"
):
    """Export an image from GEE with a given scale and area of interest
    to the Google Drive. If input data is an ImageCollection, it will convert it
    into an image and then export. The collection should contains only single data,
    for example NDVI bands or precipitation bands or LST bands.

        Args:
            ds (ee.Image|ee.ImageCollection): The input ee.Image or ee.ImageCollection
            aoi (FeatureCollection): The area of interest to clip the images.
            folder_name (str): An output file name. Default is GEE_Data
            res (int): A spatial resolution in meters. Default is 1km.
            crs (str): A crs of interest. Default to epsg:4326

        Returns:
            ee.Image: the clipped image with crs: 4326
    """
    if isinstance(aoi, (ee.geometry.Geometry, list)):
        aoi = ee.Geometry.Polygon(aoi)
    if isinstance(ds, ee.ImageCollection):
        # Convert it to an image
        img = ds.toBands()
        # get bands and rename
        oldband = img.bandNames().getInfo()
        newband = ["_".join(i.split("_")[::-1]) for i in oldband]
        # Rename it
        new_img = img.select(oldband, newband).clip(aoi)
    elif isinstance(ds, ee.Image):
        new_img = ds.clip(aoi)
    else:
        raise TypeError("Unsupported data type!")
    if isinstance(aoi, ee.featurecollection.FeatureCollection):
        aoi = aoi.geometry().bounds().getInfo()["coordinates"]
    # Initialize the task of downloading an image
    task = ee.batch.Export.image.toDrive(
        image=new_img,  # an ee.Image object.
        # an ee.Geometry object.
        region=aoi,
        description=folder_name,
        folder=folder_name,
        fileNamePrefix=file_name,
        crs=crs,
        scale=res,
        maxPixels=1e13,
    )
    task.start()


def export_img_to_asset(
    ds, aoi, assetId, description="Exported_Data_To_Asset", res=1000, crs=None
):
    """Export an image from GEE with a given scale and area of interest
    to the Google Drive. If input data is an ImageCollection, it will convert it
    into an image and then export. The collection should contains only single data,
    for example NDVI bands or precipitation bands or LST bands.

        Args:
            ds (ee.Image|ee.ImageCollection): The input ee.Image or ee.ImageCollection
            aoi (FeatureCollection): The area of interest to clip the images.
            folder_name (str): An output file name. Default is GEE_Data
            res (int): A spatial resolution in meters. Default is 1km.
            crs (str|optional): The output crs. Default to EPSG:4326

        Returns:
            ee.Image: the clipped image with crs: 4326
    """
    if crs is None:
        crs = "EPSG:4326"
    if isinstance(ds, ee.ImageCollection):
        # Convert it to an image
        img = ds.toBands()
        # get bands and rename
        oldband = img.bandNames().getInfo()
        newband = ["_".join(i.split("_")[::-1]) for i in oldband]
        # Rename it
        new_img = img.select(oldband, newband).clip(aoi)
    elif isinstance(ds, ee.Image):
        new_img = ds.clip(aoi)
    else:
        raise TypeError("Unsupported data type!")

    # Initialize the task of downloading an image
    task = ee.batch.Export.image.toAsset(
        image=new_img,  # an ee.Image object.
        # an ee.Geometry object.
        region=aoi.geometry().bounds().getInfo()["coordinates"],
        description=description,
        assetId=assetId,
        crs=crs,
        scale=res,
    )
    task.start()
