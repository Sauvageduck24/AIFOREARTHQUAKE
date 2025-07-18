#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convierte un .tif en un .gpkg con una máscara poligonal.

Modos disponibles:
  • valid     → polígonos donde el raster tiene datos (≠ nodata)
  • nodata    → polígonos donde el raster es nodata
  • threshold → polígonos donde banda > umbral
  • full      → un único polígono con la huella completa del raster
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, box
import geopandas as gpd
from scipy.ndimage import binary_dilation

def raster_to_polygons(arr, transform, mask):
    """Vectoriza la `mask` y devuelve una lista de geometrías."""
    polys = shapes(arr.astype(np.uint8), mask=mask, transform=transform)
    return [shape(geom) for geom, _ in polys]

def tif2gpkg(tif_path: Path, gpkg_out: Path, *, mode="valid",
             band=1, threshold=None, dilate=0, class_value=1):
    with rasterio.open(tif_path) as src:
        data      = src.read(band)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata

    if mode == "valid":
        mask = data != nodata if nodata is not None else np.ones_like(data, bool)

    elif mode == "nodata":
        mask = data == nodata if nodata is not None else np.zeros_like(data, bool)

    elif mode == "threshold":
        if threshold is None:
            raise ValueError("En modo 'threshold' debes pasar --threshold")
        mask = data > threshold

    elif mode == "full":
        mask = np.ones_like(data, bool)

    else:
        raise ValueError(f"Modo desconocido: {mode}")

    # Dilatación opcional para cerrar agujeros o bordes irregulares
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)

    if mode == "full":
        # Un solo polígono con la huella completa
        geom = box(*rasterio.transform.array_bounds(*data.shape, transform))
        geoms = [geom]
    else:
        geoms = raster_to_polygons(mask, transform, mask)

    if not geoms:
        raise RuntimeError("No se generaron geometrías; revisa tus parámetros")

    gdf = gpd.GeoDataFrame({"class": class_value}, geometry=geoms, crs=crs)
    gpkg_out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(gpkg_out, driver="GPKG")
    print(f"Guardado {len(gdf)} geometrías en {gpkg_out}")

def main():
    parser = argparse.ArgumentParser(description="Convierte .tif a .gpkg")
    parser.add_argument("tif", type=Path, help="Raster de entrada (.tif)")
    parser.add_argument("gpkg", type=Path, help="GeoPackage de salida (.gpkg)")
    parser.add_argument("--mode", choices=["valid", "nodata", "threshold", "full"],
                        default="valid", help="Tipo de máscara a generar")
    parser.add_argument("--band", type=int, default=1, help="Banda a usar")
    parser.add_argument("--threshold", type=float, help="Umbral numérico")
    parser.add_argument("--dilate", type=int, default=0,
                        help="Iteraciones de dilatación estructural (0 = sin)")
    parser.add_argument("--class-value", type=int, default=1,
                        help="Valor de atributo 'class' asignado")
    args = parser.parse_args()

    tif2gpkg(args.tif, args.gpkg, mode=args.mode, band=args.band,
             threshold=args.threshold, dilate=args.dilate,
             class_value=args.class_value)

if __name__ == "__main__":
    main()
