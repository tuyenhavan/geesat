import math

import ee


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


def slope_correction(
    collection,
    buffer=0,
    scale=10,
):
    """
    Apply a slope correction to the input collection using the volumetric model.
    Args:
        collection (ee.ImageCollection): Input image collection to apply the slope correction to.
        buffer (int, optional): Buffer distance in meters to apply to the layover and shadow masks. Defaults to 0.
        scale (int, optional): Scale in meters to use for the elevation data. Defaults to 10.
    Returns:
        ee.ImageCollection: Slope-corrected image collection.
    """
    dem = ee.Image("USGS/SRTMGL1_003")
    ninety_rad = ee.Image.constant(math.pi / 2)
    dem_resampling = "bilinear"

    def _volumetric_model_scf(theta_i_rad, alpha_r_rad):
        numerator = ninety_rad.subtract(theta_i_rad).add(alpha_r_rad).tan()

        denominator = ninety_rad.subtract(theta_i_rad).tan()

        return numerator.divide(denominator)

    def _erode(mask, distance):
        d = (
            mask.Not()
            .unmask(1)
            .fastDistanceTransform(30)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
        )

        return mask.updateMask(d.gt(distance))

    def _masking(alpha_r_rad, theta_i_rad, buffer_distance):

        layover = alpha_r_rad.lt(theta_i_rad)

        shadow = alpha_r_rad.gt(
            ee.Image.constant(-1).multiply(ninety_rad.subtract(theta_i_rad))
        )

        mask = layover.And(shadow)

        if buffer_distance > 0:
            mask = _erode(mask, buffer_distance)

        return mask.rename("no_data_mask")

    def _correct(image):

        band_names = image.bandNames()

        geom = image.geometry()
        proj = image.select(0).projection()

        elevation = dem.resample(dem_resampling).reproject(proj, None, scale).clip(geom)

        heading = ee.Terrain.aspect(image.select("angle")).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=1000,
            maxPixels=1e9,
        )

        heading = ee.Dictionary(heading).combine({"aspect": 0}, False).get("aspect")

        heading = ee.Algorithms.If(
            ee.Number(heading).gt(180),
            ee.Number(heading).subtract(360),
            ee.Number(heading),
        )

        # Radar geometry
        theta_i_rad = image.select("angle").multiply(math.pi / 180)

        phi_i_rad = ee.Image.constant(heading).multiply(math.pi / 180)

        # Terrain geometry
        alpha_s_rad = (
            ee.Terrain.slope(elevation).select("slope").multiply(math.pi / 180)
        )

        aspect = ee.Terrain.aspect(elevation).select("aspect").clip(geom)

        aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)

        phi_s_rad = (
            aspect.updateMask(aspect.lte(180))
            .unmask()
            .add(aspect_minus.unmask())
            .multiply(-1)
            .multiply(math.pi / 180)
        )

        # Model geometry
        phi_r_rad = phi_i_rad.subtract(phi_s_rad)

        alpha_r_rad = alpha_s_rad.tan().multiply(phi_r_rad.cos()).atan()

        alpha_az_rad = alpha_s_rad.tan().multiply(phi_r_rad.sin()).atan()

        # Gamma0
        gamma0 = image.divide(theta_i_rad.cos())

        # Volume model
        scf = _volumetric_model_scf(
            theta_i_rad,
            alpha_r_rad,
        )

        gamma0_flat = gamma0.multiply(scf)

        mask = _masking(
            alpha_r_rad,
            theta_i_rad,
            buffer,
        )

        output = ee.Image(
            gamma0_flat.updateMask(mask).rename(band_names).copyProperties(image)
        )
        output = output.addBands(
            image.select("angle"),
            None,
            True,
        )

        return output.set(
            "system:time_start",
            image.get("system:time_start"),
        )

    return collection.map(_correct)


def leefilter(image, kernel_size=9):
    """
    Apply a Lee filter to the input image.
    Args:
    image (ee.Image): Input image to apply the Lee filter to.
    kernel_size (int): Size of the kernel to use for the Lee filter.
    Returns:
    ee.Image: Image with the Lee filter applied.

    """
    bandNames = image.bandNames().remove("angle")

    # S1-GRD images are multilooked 5 times in range
    enl = 5
    # Compute the speckle standard deviation
    eta = 1.0 / math.sqrt(enl)
    eta = ee.Image.constant(eta)

    # MMSE estimator
    # Neighbourhood mean and variance
    oneImg = ee.Image.constant(1)
    # Estimate stats
    reducers = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.variance(), sharedInputs=True
    )
    stats = image.select(bandNames).reduceNeighborhood(
        reducer=reducers,
        kernel=ee.Kernel.square(kernel_size / 2, "pixels"),
        optimization="window",
    )
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat("_mean"))
    varBand = bandNames.map(lambda bandName: ee.String(bandName).cat("_variance"))

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)
    # Estimate weight
    varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(
        oneImg.add(eta.pow(2))
    )
    b = varx.divide(varz)

    # if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    output = (
        oneImg.subtract(new_b)
        .multiply(z_bar.abs())
        .add(new_b.multiply(image.select(bandNames)))
    )
    output = output.rename(bandNames)
    return image.addBands(output, None, True)


def prepare_sentinel1_collection(
    col=None,
    roi=None,
    start_date="2022-01-01",
    end_date="2022-12-31",
    polarization=None,
    orbit="ASCENDING",
):
    """
    Prepare a Sentinel-1 image collection by applying border noise correction and speckle filtering.
    Args:
        col (ee.ImageCollection, optional): Input image collection to prepare. If None, the function will select the Sentinel-1 image collection based on the provided parameters. Defaults to None.
        roi (ee.Geometry, optional): Region of interest to filter the image collection. Defaults to None.
        start_date (str, optional): Start date to filter the image collection. Defaults to "2022-01-01".
        end_date (str, optional): End date to filter the image collection. Defaults to "2022-12-31".
        polarization (str, optional): Polarization to filter the image collection. Can be "VV", "VH", or "VVVH". Defaults to None.
        orbit (str, optional): Orbit direction to filter the image collection. Can be "ASCENDING", "DESCENDING", or "BOTH". Defaults to "ASCENDING".
    Returns:
        ee.ImageCollection: Prepared Sentinel-1 image collection.
    """
    col = col if col is not None else ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
    if roi is not None:
        col = col.filterBounds(roi)
    col = col.filterDate(start_date, end_date)
    # Ensure polarization is valid
    if polarization is not None and polarization.upper() not in ["VV", "VH", "VVVH"]:
        raise ValueError("Polarization must be 'VV', 'VH', or 'VVVH'")
    # Ensure orbit is valid
    if orbit.upper() not in ["ASCENDING", "DESCENDING", "BOTH"]:
        raise ValueError("Orbit must be 'ASCENDING', 'DESCENDING', or 'BOTH'")
    orbit = orbit.upper()
    col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))
    if polarization is None:
        polarization = ["VV", "VH", "angle"]
    col = col.select(polarization)
    border_noise_correction = col.map(fmask_edge)
    speckle_filtered = border_noise_correction.map(leefilter)
    slope_corrected = slope_correction(speckle_filtered)
    # convert it dB
    db_converted = slope_corrected.map(lin_to_db)
    return db_converted
