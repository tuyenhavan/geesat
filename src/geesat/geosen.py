import math

import ee
from narwhals import col

from geesat import geogee


def mask_angle(image):
    """
    Mask out angles >= 45.23993 and <= 30.63993.
    Args:
        image (ee.Image): Image to apply the border noise masking.
    Returns:
        ee.Image: Masked image.
    """
    ang = image.select(["angle"])
    return image.updateMask(ang.lt(45.23993).And(ang.gt(30.63993))).set(
        "system:time_start", image.get("system:time_start")
    )


def mask_edge(image):
    """
    Remove edges.
    Args:
        image (ee.Image): Image to apply the border noise masking.
    Returns:
        ee.Image: Masked image.
    """
    mask = (
        image.select(0).unitScale(-25, 5).multiply(255).toByte()
    )  # .connectedComponents(ee.Kernel.rectangle(1,1), 100)
    return image.updateMask(mask.select(0)).set(
        "system:time_start", image.get("system:time_start")
    )


def lin_to_db(image):
    """
    Convert backscatter from linear to dB by removing the ratio band.
    Args:
        image (ee.Image): Image to convert
    Returns:
        ee.Image: Converted image
    """
    band_names = image.bandNames().remove("angle")
    db = (
        ee.Image.constant(10)
        .multiply(image.select(band_names).log10())
        .rename(band_names)
    )
    return image.addBands(db, None, True)


def db_to_lin(image):
    """
    Convert backscatter from dB to linear by removing the ratio band.

    Args:
        image (ee.Image): Image to convert
    Returns:
        ee.Image: Converted image
    """
    band_names = image.bandNames().remove("angle")
    lin = (
        ee.Image.constant(10)
        .pow(image.select(band_names).divide(10))
        .rename(band_names)
    )
    return image.addBands(lin, None, True)


def fmask_edge(image):
    """
    Mask out edges using the fmask band.

    Args:
    image (ee.Image): Image to apply the border noise correction to
    Returns:
    ee.Image: Corrected image
    """
    db_img = lin_to_db(image)
    result = mask_angle(db_img)
    result = mask_edge(result)
    return db_to_lin(result).set("system:time_start", image.get("system:time_start"))


def slope_correction(collection, model="volume", buffer=50):
    """
    Radiometric terrain correction for Sentinel-1.
    Adapted from this repo https://github.com/ESA-PhiLab/radiometric-slope-correction/tree/master
    Args:
    collection (ee.ImageCollection): Image collection to apply the correction to
    model (str): 'volume' or 'surface'
    buffer (int or float): Buffer distance in meters
    Returns:
    ee.ImageCollection: Terrain corrected image collection
    """

    elevation = ee.Image("USGS/SRTMGL1_003")

    ninety_rad = ee.Image.constant(math.pi / 2.0)

    # ---------------------------------------------------------------------
    # Volumetric model (Hoekman 1990)
    # ---------------------------------------------------------------------
    def volume_model(theta_i_rad, alpha_r_rad):

        numerator = ninety_rad.subtract(theta_i_rad).add(alpha_r_rad).tan()

        denominator = ninety_rad.subtract(theta_i_rad).tan()

        return numerator.divide(denominator)

    # ---------------------------------------------------------------------
    # Surface model (Ulander et al. 1996)
    # ---------------------------------------------------------------------
    def surface_model(theta_i_rad, alpha_r_rad, alpha_az_rad):

        numerator = ninety_rad.subtract(theta_i_rad).cos()

        denominator = alpha_az_rad.cos().multiply(
            ninety_rad.subtract(theta_i_rad).add(alpha_r_rad).cos()
        )

        return numerator.divide(denominator)

    # ---------------------------------------------------------------------
    # Buffer / erosion
    # ---------------------------------------------------------------------
    def erode(img, distance):

        d = (
            img.Not()
            .unmask(1)
            .fastDistanceTransform(30)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
        )

        return img.updateMask(d.gt(distance))

    # ---------------------------------------------------------------------
    # Layover-shadow mask
    # ---------------------------------------------------------------------
    def masking(alpha_r_rad, theta_i_rad, proj, buffer_distance):

        layover = alpha_r_rad.lt(theta_i_rad).rename("layover")

        shadow = alpha_r_rad.gt(
            ee.Image.constant(-1).multiply(ninety_rad.subtract(theta_i_rad))
        ).rename("shadow")

        mask = layover.And(shadow)

        if buffer_distance > 0:
            mask = erode(mask, buffer_distance)

        return mask.rename("no_data_mask")

    # ---------------------------------------------------------------------
    # Image correction
    # ---------------------------------------------------------------------
    def correct(image):

        geom = image.geometry()

        proj = image.select(1).projection()

        # Mean radar heading
        heading = ee.Number(
            ee.Terrain.aspect(image.select("angle"))
            .reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom, scale=1000, maxPixels=1e9
            )
            .get("aspect")
        )

        # Sigma0 dB -> power
        sigma0_pow = ee.Image.constant(10).pow(image.divide(10.0))

        # Radar geometry
        theta_i_rad = image.select("angle").multiply(math.pi / 180.0).clip(geom)

        phi_i_rad = ee.Image.constant(heading).multiply(math.pi / 180.0)

        # Terrain geometry
        alpha_s_rad = (
            ee.Terrain.slope(elevation)
            .select("slope")
            .multiply(math.pi / 180.0)
            .setDefaultProjection(proj)
            .clip(geom)
        )

        phi_s_rad = (
            ee.Terrain.aspect(elevation)
            .select("aspect")
            .multiply(math.pi / 180.0)
            .setDefaultProjection(proj)
            .clip(geom)
        )

        # Relative geometry
        phi_r_rad = phi_i_rad.subtract(phi_s_rad)

        # Range slope
        alpha_r_rad = alpha_s_rad.tan().multiply(phi_r_rad.cos()).atan()

        # Azimuth slope
        alpha_az_rad = alpha_s_rad.tan().multiply(phi_r_rad.sin()).atan()

        # Gamma0
        gamma0 = sigma0_pow.divide(theta_i_rad.cos())

        # Terrain correction model
        if model == "volume":

            corr_model = volume_model(theta_i_rad, alpha_r_rad)

        elif model == "surface":

            corr_model = surface_model(theta_i_rad, alpha_r_rad, alpha_az_rad)

        else:
            raise ValueError("model must be 'volume' or 'surface'")

        # Flattened Gamma0
        gamma0_flat = gamma0.divide(corr_model)

        # Back to dB
        sar_bands = image.bandNames().remove("angle")
        gamma0_flat_db = (
            ee.Image.constant(10).multiply(gamma0_flat.log10()).select(sar_bands)
        )

        # Layover / shadow mask
        mask = masking(alpha_r_rad, theta_i_rad, proj, buffer)

        return gamma0_flat_db.addBands(mask).copyProperties(
            image, image.propertyNames()
        )

    return collection.map(correct)


def leefilter(image, kernel_size=7):
    """
    Apply a Lee filter to the input image.
    Args:
    image (ee.Image): Input image to apply the Lee filter to.
    kernel_size (int): Size of the kernel to use for the Lee filter.
    Returns:
    ee.Image: Image with the Lee filter applied.

    """
    band_names = image.bandNames().remove("angle")

    # S1-GRD images are multilooked 5 times in range
    enl = 5
    # Compute the speckle standard deviation
    eta = 1.0 / math.sqrt(enl)
    eta = ee.Image.constant(eta)

    # MMSE estimator
    # Neighbourhood mean and variance
    one_img = ee.Image.constant(1)
    # Estimate stats
    reducers = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.variance(), sharedInputs=True
    )
    stats = image.select(band_names).reduceNeighborhood(
        reducer=reducers,
        kernel=ee.Kernel.square(kernel_size / 2, "pixels"),
        optimization="window",
    )
    mean_band = band_names.map(lambda bandName: ee.String(bandName).cat("_mean"))
    var_band = band_names.map(lambda bandName: ee.String(bandName).cat("_variance"))

    z_bar = stats.select(mean_band)
    varz = stats.select(var_band)
    # Estimate weight
    varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(
        one_img.add(eta.pow(2))
    )
    b = varx.divide(varz)

    # if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    output = (
        one_img.subtract(new_b)
        .multiply(z_bar.abs())
        .add(new_b.multiply(image.select(band_names)))
    )
    output = output.rename(band_names)
    return image.addBands(output, None, True)


def prepare_sentinel1_collection(
    roi,
    orbit_pass="ASCENDING",
    start_date="2022-01-01",
    end_date="2022-12-31",
    buffer=50,
    model="volume",
):
    """Prepare a Sentinel-1 image collection by applying the Lee filter and slope correction.
    Args:
        roi (ee.Geometry): Region of interest to filter the image collection.
        polarization (str, optional): Polarization to filter the image collection. Defaults to 'VV'.
        orbit_pass (str, optional): Orbit pass to filter the image collection. Either 'ASCENDING' or 'DESCENDING'. Defaults to 'ASCENDING'.
        start_date (str, optional): Start date for filtering the image collection. Defaults to '2022-01-01'.
        end_date (str, optional): End date for filtering the image collection. Defaults to '2022-12-31'.
        buffer (int, optional): Buffer distance in meters for slope correction. Defaults to 50.
        model (str, optional): Slope correction model to use. Either 'volume' or 'surface'. Defaults to 'volume'.
    Returns:
        ee.ImageCollection: Prepared Sentinel-1 image collection.
    """

    col = ee.ImageCollection("COPERNICUS/S1_GRD")
    orbit_pass = orbit_pass.upper()
    # check if the orbit pass is valid
    if orbit_pass not in ["ASCENDING", "DESCENDING", "BOTH"]:
        raise ValueError(
            "orbit_pass must be either 'ASCENDING', 'DESCENDING', or 'BOTH'"
        )
    if orbit_pass == "BOTH":
        orbit_pass = ["ASCENDING", "DESCENDING"]
    else:
        orbit_pass = [orbit_pass]
    # check if the model is valid
    if model not in ["volume", "surface"]:
        raise ValueError("model must be either 'volume' or 'surface'")
    col = (
        col.filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.inList("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    col = col.map(db_to_lin).map(leefilter).map(lin_to_db)
    col = slope_correction(col, model=model, buffer=buffer)
    return col.select(["VV", "VH"])


def generate_water_occurrence(
    roi,
    collection=None,
    start_date="2022-01-01",
    end_date="2022-12-31",
    buffer=50,
    model="volume",
    polarization="VV",
    water_threshold=-15,
    orbit_pass="ASCENDING",
):
    """Generate a water occurrence map from Sentinel-1 image collection.
    Args:
        collection (ee.ImageCollection, optional): Sentinel-1 image collection to generate the water occurrence map from. Defaults to None (ee.ImageCollection("COPERNICUS/S1_GRD").
        roi (ee.Geometry): Region of interest to filter the image collection.
        start_date (str, optional): Start date for filtering the image collection. Defaults to '2022-01-01'.
        end_date (str, optional): End date for filtering the image collection. Defaults to '2022-12-31'.
        buffer (int, optional): Buffer distance in meters for slope correction. Defaults to 50.
        model (str, optional): Slope correction model to use. Either 'volume' or 'surface'. Defaults to 'volume'.
        polarization (str, optional): Polarization to use for water detection. Defaults to 'VV'.
        water_threshold (float, optional): Threshold for water detection in dB. Defaults to -15.
    Returns:
        ee.Image: Water occurrence map.
    """
    # check polarization
    if polarization not in ["VV", "VH"]:
        raise ValueError("polarization must be either 'VV' or 'VH'")
    if collection is None:
        collection = prepare_sentinel1_collection(
            roi,
            start_date=start_date,
            end_date=end_date,
            buffer=buffer,
            model=model,
            orbit_pass=orbit_pass,
        )
    else:
        if orbit_pass == "BOTH":
            collection = (
                collection.filterBounds(roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(
                    ee.Filter.inList(
                        "orbitProperties_pass", ["ASCENDING", "DESCENDING"]
                    )
                )
            )
        else:
            collection = (
                collection.filterBounds(roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
            )
    collection = geogee.generate_monthly_composite(
        collection, aggregate_method="median"
    )
    water_mask = (
        collection.map(lambda img: img.select(polarization).lt(water_threshold))
        .sum()
        .divide(collection.size())
        .multiply(100)
    )
    return water_mask.rename("water_occurrence")
