import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# RUTAS DE PRUEBA: AJUSTA ESTO A TU CASO
gpkg_path = Path("data/international-charter-eo4ai-etq-challenge/Training/Annotations/Call_1075_China/china_training.gpkg")
raster_path = Path("data/international-charter-eo4ai-etq-challenge/Training/Call_1075_China/WV03N28_549166E087_4022222023062600000000MS00-calibrated/Optical_Calibration/r-blue.tif")  # Cambia a una banda real

# Leer GPKG
gdf = gpd.read_file(gpkg_path)
print("GPKG CRS:", gdf.crs)
print("GPKG bounds:", gdf.total_bounds)
print("GPKG rows:", len(gdf))

# Leer raster de referencia
with rasterio.open(raster_path) as src:
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)
    transform = src.transform
    shape = (src.height, src.width)

# Reproyectar geometrías si es necesario
if gdf.crs != src.crs:
    gdf = gdf.to_crs(src.crs)
    print("GPKG reproyectado a:", gdf.crs)
