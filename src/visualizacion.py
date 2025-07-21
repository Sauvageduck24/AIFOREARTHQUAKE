import os
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from preprocess import process_event
import rasterio
from pathlib import Path
from shapely.geometry import shape, box
import json
import re
from datetime import datetime, timezone
import numpy as np
from skimage.transform import resize
from typing import Tuple, List, Dict, Union

def get_matching_gdf(
    gpkg_path: str,
    raster_path: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Devuelve:
      a) gdf_img_crs  -> GeoDataFrame en el CRS de raster_path
      b) gdf_wgs84    -> El mismo GeoDataFrame en WGS-84 (EPSG:4326)
    """
    # --- Leer GPKG ---
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError(f"{gpkg_path} no contiene CRS.")
    
    # --- Leer raster sólo para coger su CRS ---
    with rasterio.open(raster_path) as src:
        img_crs = src.crs
    
    # --- Reproyectar si fuese necesario ---
    if gdf.crs != img_crs:
        gdf_img_crs = gdf.to_crs(img_crs)
    else:
        gdf_img_crs = gdf.copy()
    
    # GeoDataFrame en WGS-84 (para STAC, bounding‐box, etc.)
    if gdf.crs != 4326:
        gdf_wgs84 = gdf.to_crs(4326)
    else:
        gdf_wgs84 = gdf.copy()
    
    return gdf_img_crs, gdf_wgs84


def print_crs_info(gdf_original: gpd.GeoDataFrame,
                   gdf_img_crs: gpd.GeoDataFrame,
                   raster_path: str) -> None:
    """
    Imprime por pantalla los CRS de la fuente vectorial y el raster,
    indicando si se ha reproyectado.
    """
    with rasterio.open(raster_path) as src:
        img_crs = src.crs
    
    print(f"  ▸ CRS raster      : {img_crs}")
    print(f"  ▸ CRS gdf original: {gdf_original.crs}")
    if gdf_original.crs != img_crs:
        print(f"    ⇢  Reproyectado gdf → {img_crs}")
    else:
        print("    ↳ CRS ya coinciden.")


# Configura la visualización
plt.rcParams['figure.figsize'] = (10, 10)

# Rutas de archivos
adiyaman_gpkg = r"E:\Sauvageduck24\AIFOREARTHQUAKE\data\international-charter-eo4ai-etq-challenge\Training\Annotations\Call_919_Turkey\adiyaman_training.gpkg"
antakya_gpkg = r"E:\Sauvageduck24\AIFOREARTHQUAKE\data\international-charter-eo4ai-etq-challenge\Training\Annotations\Call_919_Turkey\antakya_east_training.gpkg"

adiyaman_img_dir = r"E:\Sauvageduck24\AIFOREARTHQUAKE\data\international-charter-eo4ai-etq-challenge\Training\Call_919_Turkey"
antakya_img_dir = r"E:\Sauvageduck24\AIFOREARTHQUAKE\data\international-charter-eo4ai-etq-challenge\Training\Call_919_Turkey"

# Cargar el GeoDataFrame
print("Cargando geometrías de Adıyaman...")
gdf_adiyaman = gpd.read_file(adiyaman_gpkg)
print(f"Total de geometrías: {len(gdf_adiyaman)}")
print("CRS:", gdf_adiyaman.crs)
# head
print(gdf_adiyaman.head())

print("Cargando geometrías de Antakya...")
gdf_antakya = gpd.read_file(antakya_gpkg)
print(f"Total de geometrías: {len(gdf_antakya)}")
print("CRS:", gdf_antakya.crs)
# head
print(gdf_antakya.head())

# Visualizar algunas geometrías
fig, ax = plt.subplots()
gdf_adiyaman.plot(column='damaged', cmap='coolwarm', legend=True, edgecolor='black', ax=ax)
plt.title("Distribución de edificios en Adıyaman")
plt.show()

fig, ax = plt.subplots()
gdf_antakya.plot(column='damaged', cmap='coolwarm', legend=True, edgecolor='black', ax=ax)
plt.title("Distribución de edificios en Antakya")
plt.show()

ACTIVATION_DATE = datetime.fromisoformat("2023-02-06").replace(tzinfo=timezone.utc)   # Charter Turkey

def extract_date_from_name(name: str) -> datetime | None:
    m = re.search(r"(\d{8})", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    return None

def find_tifs_for_gpkg(
    gpkg_path: str | Path,
    training_root: str | Path = r"E:\Sauvageduck24\AIFOREARTHQUAKE\data\international-charter-eo4ai-etq-challenge\Training\Call_919_Turkey",
):
    gpkg_path = Path(gpkg_path)
    training_root = Path(training_root)
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError("El .gpkg no tiene CRS definido.")
    gdf = gdf.to_crs(4326)
    aoi = gdf.unary_union  # MultiPolygon de toda el área
    resultados = []
    for item_json in training_root.glob("*-calibrated/*.json"):
        try:
            with open(item_json, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception as e:
            print(f"[WARN] No se pudo leer {item_json}: {e}")
            continue
        if "geometry" in item:
            escena_geom = shape(item["geometry"])
        elif "bbox" in item:
            bbox = item["bbox"]
            escena_geom = box(*bbox)
        else:
            print(f"[WARN] {item_json} sin geometry ni bbox")
            continue
        if not escena_geom.intersects(aoi):
            continue
        calib_dir = item_json.parent / "Optical_Calibration"
        tifs = sorted(calib_dir.glob("r-*.tif"))
        if not tifs:
            print(f"[WARN] Sin TIFFs en {calib_dir}")
            continue
        scene_date = None
        scene_date_str = item.get("properties", {}).get("datetime")
        if scene_date_str:
            try:
                scene_date = datetime.fromisoformat(scene_date_str.replace("Z", "+00:00"))
            except Exception:
                scene_date = None
        if scene_date is None:
            scene_date = extract_date_from_name(item_json.stem)
        relation = None
        if scene_date:
            if scene_date.tzinfo is None:
                scene_date = scene_date.replace(tzinfo=timezone.utc)
            relation = "post" if scene_date >= ACTIVATION_DATE else "pre"
        resultados.append(
            dict(
                scene_id=item_json.parent.name,
                scene_date=scene_date.isoformat() if scene_date else "unknown",
                relation=relation,
                item_json=str(item_json),
                tifs=[str(p) for p in tifs],
            )
        )
    return resultados

# Cambia el path al gpkg que quieras analizar:
print("\n","-"*50,"\n")

print("Buscando tifs para Adiyaman...")
gpkg = adiyaman_gpkg
matches = find_tifs_for_gpkg(gpkg)

if not matches:
    print("No se encontraron escenas que cubran el AOI.")
else:
    for m in matches:
        print(f"\n=== {m['scene_id']}  ({m['relation']}) ===")
        print(f"Fecha escena : {m['scene_date']}")
        print("TIFFs:")
        for p in m["tifs"]:
            print("   ", p)
            # Mostrar información del TIFF
            try:
                with rasterio.open(p) as src:
                    print(f"      CRS: {src.crs}")
                    print(f"      Tamaño: {src.width} x {src.height}")
                    print(f"      Número de bandas: {src.count}")
                    print(f"      Resolución espacial: {src.res}")
                    print(f"      Tipo de dato: {src.dtypes}")
                # Mostrar info de CRS y reproyección usando las funciones nuevas
                gdf_img_crs, gdf_wgs84 = get_matching_gdf(adiyaman_gpkg, p)
                print_crs_info(gpd.read_file(adiyaman_gpkg), gdf_img_crs, p)
                # --- Combinar imagen y gdf reproyectado (recorte por AOI) ---
                geoms = [geom for geom in gdf_img_crs.geometry if geom.is_valid and not geom.is_empty]
                if geoms:
                    with rasterio.open(p) as src:
                        out_image, out_transform = mask(src, geoms, crop=True)
                        # Visualiza el recorte (solo la primera banda)
                        plt.figure(figsize=(6, 6))
                        plt.imshow(out_image[0], cmap='gray')
                        plt.title("Imagen recortada por AOI/edificios (Adiyaman)")
                        plt.axis('off')
                        plt.show()
                else:
                    print("No hay geometrías válidas para recortar.")
            except Exception as e:
                print(f"      [ERROR] No se pudo leer el TIFF: {e}")

print("\n","-"*50,"\n")

print("Buscando tifs para Antakya...")
gpkg = antakya_gpkg
matches = find_tifs_for_gpkg(gpkg)

if not matches:
    print("No se encontraron escenas que cubran el AOI.")
else:
    for m in matches:
        print(f"\n=== {m['scene_id']}  ({m['relation']}) ===")
        print(f"Fecha escena : {m['scene_date']}")
        print("TIFFs:")
        for p in m["tifs"]:
            print("   ", p)
            # Mostrar información del TIFF
            try:
                with rasterio.open(p) as src:
                    print(f"      CRS: {src.crs}")
                    print(f"      Tamaño: {src.width} x {src.height}")
                    print(f"      Número de bandas: {src.count}")
                    print(f"      Resolución espacial: {src.res}")
                    print(f"      Tipo de dato: {src.dtypes}")
                # Mostrar info de CRS y reproyección usando las funciones nuevas
                gdf_img_crs, gdf_wgs84 = get_matching_gdf(antakya_gpkg, p)
                print_crs_info(gpd.read_file(antakya_gpkg), gdf_img_crs, p)
                # --- Combinar imagen y gdf reproyectado (recorte por AOI) ---
                geoms = [geom for geom in gdf_img_crs.geometry if geom.is_valid and not geom.is_empty]
                if geoms:
                    with rasterio.open(p) as src:
                        out_image, out_transform = mask(src, geoms, crop=True)
                        # Visualiza el recorte (solo la primera banda)
                        plt.figure(figsize=(6, 6))
                        plt.imshow(out_image[0], cmap='gray')
                        plt.title("Imagen recortada por AOI/edificios (Antakya)")
                        plt.axis('off')
                        plt.show()
                else:
                    print("No hay geometrías válidas para recortar.")
            except Exception as e:
                print(f"      [ERROR] No se pudo leer el TIFF: {e}")