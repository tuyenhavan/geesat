# GeeSat 🛰️

A Python package for satellite data processing and analysis using Google Earth Engine (GEE). GeeSat provides tools for vegetation indices calculation, cloud masking, data extraction, and satellite image processing.

## ✨ Features

- **Vegetation Indices**: NDVI, EVI, NDWI, SAVI calculation
- **Cloud Masking**: Sentinel-2, Landsat, and MODIS cloud removal
- **Data Processing**: Raster extraction, temperature conversion, data scaling
- **Export Functions**: Google Drive and GEE Asset export
- **Utilities**: Date generation, GeoPandas conversion, format handling

## 🚀 Installation

### Prerequisites
- Python >= 3.12
- Google Earth Engine account and authentication

### Install
```bash
git clone https://github.com/yourusername/geesat.git
cd geesat
pip install -e .
```

### Dependencies
- `earthengine-api>=1.6.11`
- `geopandas>=1.1.1`
- `python-dateutil>=2.9.0`

## 🎯 Quick Start

```python
import ee
from geesat import geogee, common

# Initialize Earth Engine
ee.Initialize()

# Apply cloud masking to Sentinel-2 data
aoi = ee.Geometry.Point([10.48, 51.80])
masked_collection = geogee.sen2_cloud_mask(
    aoi, '2023-01-01', '2023-12-31', cloud_filter=20
)
```

## 📦 Main Functions

### Vegetation Indices (`geogee.py`)
```python
geogee.calculate_ndvi(image)      # NDVI calculation
geogee.calculate_evi(image)       # Enhanced Vegetation Index
geogee.calculate_ndwi(image)      # Water Index
geogee.calculate_savi(image)      # Soil Adjusted VI
```

### Cloud Masking (`geogee.py`)
```python
geogee.sen2_cloud_mask(aoi, start, end)     # Sentinel-2 masking
geogee.landsat_cloud_mask(collection)       # Landsat masking
geogee.modis_cloud_mask(col, from_bit, to_bit)  # MODIS masking
```

### Data Processing (`geogee.py`)
```python
geogee.extract_raster_values_by_polygon(image, polygons)
geogee.convert_landsat_lst_to_celsius(collection)
geogee.kelvin_to_celsius(data)
geogee.scale_data(data, factor)
```

### Utilities (`common.py`)
```python
common.gdf_to_ee(gdf)                    # GeoPandas to EE conversion
common.daily_date_list(year, month, day) # Date sequence generation
common.geedate_to_python_datetime(code)  # Timestamp conversion
```

## 🙋‍♂️ Contact

- **Author:** tuyenrss
- **Email:** tuyenmassey@gmail.com

---

⭐ If you find GeeSat useful, please give it a star!