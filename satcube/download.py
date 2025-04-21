import ee
import cubexpress
import pathlib
from typing import Optional
from datetime import datetime

def download_data(
    lon: float,
    lat: float,
    cs_cdf: Optional[float] = 0.6,
    buffer_size: Optional[int] = 1280,
    start_date: Optional[str] = "2015-01-01",
    end_date: Optional[str] = datetime.today().strftime('%Y-%m-%d'),
    outfolder: Optional[str] = "raw/"
) -> pathlib.Path:
    """
    Download Sentinel-2 imagery data using cubexpress and Earth Engine API.

    Args:
        lon (float): Longitude of the point of interest.
        lat (float): Latitude of the point of interest.
        cs_cdf (Optional[float]): Cloud mask threshold (default 0.6).
        buffer_size (Optional[int]): Buffer size for image extraction (default 1280).
        start_date (Optional[str]): Start date for image filtering (default "2015-01-01").
        end_date (Optional[str]): End date for image filtering (default today’s date).
        outfolder (Optional[str]): Output folder to save images (default "raw/").

    Returns:
        pathlib.Path: Path to the folder where the data is stored.
    """
    
    # Initialize Earth Engine
    ee.Initialize(project="ee-julius013199")

    # Define point of interest
    point = ee.Geometry.Point([lon, lat])
    
    # Filter image collection by location and date
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(point) \
                    .filterDate(start_date, end_date)

    # Get image IDs
    image_ids = collection.aggregate_array('system:id').getInfo()
    
    # Cloud mask function
    def cloud_mask(image) -> ee.Image:
        """Apply cloud mask to the image."""
        return image.select('MSK_CLDPRB').lt(20)

    # Apply cloud mask
    collection = collection.map(cloud_mask)
    
    # Generate geotransform for cubexpress
    geotransform = cubexpress.lonlat2rt(lon=lon, lat=lat, edge_size=buffer_size, scale=10)

    # Prepare requests for cubexpress
    requests = [
        cubexpress.Request(
            id=f"s2test_{i}",
            raster_transform=geotransform,
            bands=["B4", "B3", "B2"],  # RGB bands
            image=ee.Image(image_id).divide(10000)  # Adjust image scaling
        )
        for i, image_id in enumerate(image_ids)
    ]
    
    # Create request set
    cube_requests = cubexpress.RequestSet(requestset=requests)

    # Set output folder
    output_path = pathlib.Path(outfolder)

    # Download the data
    cubexpress.getcube(
        request=cube_requests,
        output_path=output_path,
        nworkers=4,
        max_deep_level=5
    )
    
    return output_path
