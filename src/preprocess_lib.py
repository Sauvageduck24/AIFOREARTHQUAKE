import logging
import math
import os
import re
import shutil
import signal
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer
from shapely import wkb
from shapely.geometry import box, mapping, shape
from skimage.registration import phase_cross_correlation
import json
from skimage.transform import resize
from tqdm import tqdm
import fiona
from shapely.ops import transform as sxf
from osgeo import gdal, osr
import skimage.draw
import statistics
from scipy.ndimage import shift as ndi_shift

# Configuración global
warnings.filterwarnings("ignore", category=UserWarning)
ACTIVATION_DATES = {
    "Call_919_Turkey": datetime(2023, 2, 6, tzinfo=timezone.utc),
    "Call_1095_Myanmar": datetime(2025, 3, 23, tzinfo=timezone.utc),
    "Call_1075_China": datetime(2025, 1, 8, tzinfo=timezone.utc),
    "Call_920_Syria": datetime(2023, 2, 9, tzinfo=timezone.utc),
    "Call_965_Morocco": datetime(2023, 9, 10, tzinfo=timezone.utc),
    "Call_970_Afghanistan": datetime(2023, 10, 9, tzinfo=timezone.utc)
}

TILE_SIZE = 224
BUFFER_METERS = 700  # Buffer consistente en metros
NO_UPSAMPLE = True  # evita escalar hacia arriba
MAX_EXTENT_OVERLAP_HRS = 48  # acepta escenas +48 h fuera del extent.temporal.end

# === GLOBAL PERCENTILES ===
PERCENTILE_SAMPLE_SIZE = 1000
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98
P = None  # Se calculará al inicio

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("preprocess")

# --- All functions from preprocess_old.py except main() and CLI logic ---
def find_fid_column(gdf):
    fid_cols = [col for col in gdf.columns if col.lower().startswith('fid_')]
    if fid_cols:
        return fid_cols[0]
    if 'id' in gdf.columns:
        return 'id'
    return None

def load_gpkg_with_fiona(gpkg_path):
    features = []
    feature_ids = []
    selected_fid_column = None
    crs = None
    with fiona.open(gpkg_path) as src:
        crs = src.crs
        # print(f"[CHECK] gdf.crs: {crs}")  # Checklist 1 (comentado)
        schema_columns = list(src.schema['properties'].keys())
        logger.debug(f"Columnas disponibles en el esquema: {schema_columns}")
        fid_columns = [col for col in schema_columns if col.lower().startswith('fid_')]
        if fid_columns:
            selected_fid_column = fid_columns[0]
            logger.info(f"Encontrada columna FID específica: {selected_fid_column}")
            for feature in src:
                features.append(feature)
                fid_value = feature['properties'].get(selected_fid_column)
                feature_ids.append(fid_value)
        else:
            logger.info("No se encontraron columnas fid_*, usando ID de Fiona")
            for feature in src:
                features.append(feature)
                feature_ids.append(feature.get('id'))
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    if selected_fid_column:
        gdf[selected_fid_column] = feature_ids
        logger.info(f"Asignados {len(feature_ids)} IDs a la columna '{selected_fid_column}'")
    else:
        gdf['id'] = feature_ids
        logger.info(f"Asignados {len(feature_ids)} IDs a la columna 'id'")
    if gdf.crs is None:
        raise ValueError(f"El archivo {gpkg_path} no tiene CRS definido.")
    return gdf

def extract_date_from_name(name):
    m = re.search(r"(\d{8})", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    return None

def get_activation_date(call_dir: Path) -> datetime | None:
    coll_json = call_dir / "collection.json"
    if coll_json.exists():
        with open(coll_json, "r", encoding="utf-8") as f:
            txt = f.read()
        pat = re.search(r"Date of Charter Activation[:\s]*(\d{4}-\d{2}-\d{2})", txt)
        if pat:
            try:
                dt = datetime.fromisoformat(pat.group(1)).replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
    dt = ACTIVATION_DATES.get(call_dir.name)
    if dt:
        return dt
    logger.warning(f"No activation date for {call_dir.name}")
    return None

GSD_CACHE = {}
GSD_RAW_CACHE = {}

def read_gsd(item_json):
    with open(item_json) as f:
        item = json.load(f)
    gsd = item['properties'].get('gsd')
    def get_gsd_from_tifs(calib_dir):
        band_files = list(calib_dir.glob('r-*.tif'))
        gsd_values = []
        for tif in band_files:
            tif_path = str(tif)
            if tif_path in GSD_CACHE:
                val = GSD_CACHE[tif_path]
                if val is not None and 0.1 <= val <= 100:
                    gsd_values.append(val)
                continue
            try:
                ds = gdal.Open(tif_path)
                if ds is not None:
                    gt = ds.GetGeoTransform()
                    val = abs(gt[1])
                    if 0.1 <= val <= 100:
                        gsd_values.append(val)
                        GSD_CACHE[tif_path] = val
                    else:
                        GSD_CACHE[tif_path] = None
                    ds = None
            except Exception as e:
                logger.warning(f"No se pudo leer GSD de {tif}: {e}")
                GSD_CACHE[tif_path] = None
        if gsd_values:
            try:
                import statistics
                return statistics.mode(gsd_values)
            except statistics.StatisticsError:
                return float(np.mean(gsd_values))
        return None
    if gsd is None or not (0.1 <= gsd <= 100):
        calib_dir = item_json.parent / "Optical_Calibration"
        gsd_from_tif = get_gsd_from_tifs(calib_dir)
        if gsd_from_tif is not None:
            gsd = gsd_from_tif
        else:
            logger.warning(f"No se pudo obtener un GSD válido para {item_json}")
            gsd = None
    return gsd

def find_scenes_for_aoi(aoi_geom, training_root, interval_start=None, interval_end=None):
    """
    Para cada fecha base, combina los canales de MS00-calibrated (multiespectrales) y PN00-calibrated (pan) en una sola escena.
    Si solo existe una de las dos, usa lo que haya. Si existe una carpeta que no es ni MS ni PN, usa todos los canales que tenga.
    Así, cada escena tendrá el máximo de bandas posibles para esa fecha.
    """
    from pathlib import Path
    import json
    from shapely.geometry import shape, box
    import geopandas as gpd
    from datetime import datetime, timedelta, timezone
    import re
    scenes = []
    training_root = Path(training_root)
    # 1. Agrupar carpetas por fecha base
    carpetas = list(training_root.glob("*-calibrated"))
    escenas_por_fecha = {}
    regex_fecha = re.compile(r"(.+?)(?:_MS00-calibrated|_PN00-calibrated|-calibrated)$")
    for carpeta in carpetas:
        m = regex_fecha.match(carpeta.name)
        if not m:
            continue
        fecha_base = m.group(1)
        if fecha_base not in escenas_por_fecha:
            escenas_por_fecha[fecha_base] = {}
        if carpeta.name.endswith("MS00-calibrated"):
            escenas_por_fecha[fecha_base]["MS"] = carpeta
        elif carpeta.name.endswith("PN00-calibrated"):
            escenas_por_fecha[fecha_base]["PN"] = carpeta
        else:
            escenas_por_fecha[fecha_base]["OTRO"] = carpeta
    # 2. Para cada fecha base, combinar bandas
    for fecha_base, paths in escenas_por_fecha.items():
        bands = {}
        scene_geom = None
        scene_date = None
        properties = {}
        assets = {}
        # MS: añadir RGB, NIR, etc.
        if "MS" in paths:
            ms_dir = paths["MS"]
            ms_jsons = list(ms_dir.glob("*.json"))
            if ms_jsons:
                with open(ms_jsons[0], 'r') as f:
                    item = json.load(f)
                if 'geometry' in item:
                    scene_geom = shape(item['geometry'])
                    scene_geom = gpd.GeoSeries([scene_geom], crs=4326)[0]
                elif 'bbox' in item:
                    scene_geom = box(*item['bbox'])
                    scene_geom = gpd.GeoSeries([scene_geom], crs=4326)[0]
                scene_date = item.get('properties', {}).get('datetime')
                if scene_date:
                    try:
                        scene_date = datetime.fromisoformat(scene_date.replace('Z', '+00:00'))
                    except:
                        from preprocess_lib import extract_date_from_name
                        scene_date = extract_date_from_name(ms_jsons[0].stem)
                properties = item.get('properties', {})
                assets = item.get('assets', {})
            calib_dir = ms_dir / "Optical_Calibration"
            # Usar assets si existen
            if assets:
                for band_name, asset in assets.items():
                    href = asset.get('href')
                    if href and not band_name.lower().startswith('pan'):
                        bands[band_name] = (ms_dir / href).resolve()
            else:
                for band_file in calib_dir.glob("r-*.tif"):
                    band_name = band_file.stem.split('-')[-1]
                    if band_name != 'pan':
                        bands[band_name] = band_file
        # PN: añadir solo pan
        if "PN" in paths:
            pn_dir = paths["PN"]
            pn_jsons = list(pn_dir.glob("*.json"))
            if not scene_geom and pn_jsons:
                with open(pn_jsons[0], 'r') as f:
                    item = json.load(f)
                if 'geometry' in item:
                    scene_geom = shape(item['geometry'])
                    scene_geom = gpd.GeoSeries([scene_geom], crs=4326)[0]
                elif 'bbox' in item:
                    scene_geom = box(*item['bbox'])
                    scene_geom = gpd.GeoSeries([scene_geom], crs=4326)[0]
                if not scene_date:
                    scene_date = item.get('properties', {}).get('datetime')
                    if scene_date:
                        try:
                            scene_date = datetime.fromisoformat(scene_date.replace('Z', '+00:00'))
                        except:
                            from preprocess_lib import extract_date_from_name
                            scene_date = extract_date_from_name(pn_jsons[0].stem)
                if not properties:
                    properties = item.get('properties', {})
                if not assets:
                    assets = item.get('assets', {})
            calib_dir = pn_dir / "Optical_Calibration"
            # Usar assets si existen
            if assets and 'pan' in assets:
                href = assets['pan'].get('href')
                if href:
                    bands['pan'] = (pn_dir / href).resolve()
            else:
                pan_file = calib_dir / "r-pan.tif"
                if pan_file.exists():
                    bands['pan'] = pan_file
        # OTRO: añadir lo que haya
        if "OTRO" in paths:
            otro_dir = paths["OTRO"]
            otro_jsons = list(otro_dir.glob("*.json"))
            if not scene_geom and otro_jsons:
                with open(otro_jsons[0], 'r') as f:
                    item = json.load(f)
                if 'geometry' in item:
                    scene_geom = shape(item['geometry'])
                    scene_geom = gpd.GeoSeries([scene_geom], crs=4326)[0]
                elif 'bbox' in item:
                    scene_geom = box(*item['bbox'])
                    scene_geom = gpd.GeoSeries([scene_geom], crs=4326)[0]
                if not scene_date:
                    scene_date = item.get('properties', {}).get('datetime')
                    if scene_date:
                        try:
                            scene_date = datetime.fromisoformat(scene_date.replace('Z', '+00:00'))
                        except:
                            from preprocess_lib import extract_date_from_name
                            scene_date = extract_date_from_name(otro_jsons[0].stem)
                if not properties:
                    properties = item.get('properties', {})
                if not assets:
                    assets = item.get('assets', {})
            calib_dir = otro_dir / "Optical_Calibration"
            # Usar assets si existen
            if assets:
                for band_name, asset in assets.items():
                    href = asset.get('href')
                    if href:
                        bands[band_name] = (otro_dir / href).resolve()
            else:
                for band_file in calib_dir.glob("r-*.tif"):
                    band_name = band_file.stem.split('-')[-1]
                    bands[band_name] = band_file
        # Si no hay bandas, saltar
        if not bands or scene_geom is None or scene_date is None:
            continue
        # Validar intersección con AOI
        if not aoi_geom.intersects(scene_geom):
            continue
        # Filtrar por intervalo temporal si aplica
        if interval_start and interval_end and scene_date:
            if scene_date < interval_start:
                continue
            if scene_date > interval_end + timedelta(hours=MAX_EXTENT_OVERLAP_HRS):
                logger.warning(f"{fecha_base} fuera de intervalo ({scene_date} > {interval_end})  — descartado")
                continue
            elif scene_date > interval_end:
                hours_over = (scene_date - interval_end).total_seconds() / 3600
                logger.info(f"{fecha_base} {scene_date} a {hours_over:.1f} h del límite; lo acepto por margen +{MAX_EXTENT_OVERLAP_HRS} h")
        # Leer GSD
        gsd = None
        gsd_raw = None
        if 'gsd' in properties:
            gsd_raw = properties.get('gsd')
            print(f'[RAW GSD] {fecha_base} | gsd extraído de properties: {gsd_raw}')
            GSD_RAW_CACHE[fecha_base] = gsd_raw
            gsd = gsd_raw
        # Si no hay GSD, intentar leerlo de los tifs
        if gsd is None or not (0.1 <= float(gsd) <= 100):
            # Buscar en MS, luego PN, luego OTRO
            for key in ['MS', 'PN', 'OTRO']:
                if key in paths:
                    from preprocess_lib import read_gsd
                    jsons = list(paths[key].glob("*.json"))
                    if jsons:
                        gsd = read_gsd(jsons[0])
                        if gsd is not None:
                            break
        scene = {
            'scene_id': fecha_base,
            'date': scene_date,
            'geometry': scene_geom,
            'path': paths.get('MS') or paths.get('PN') or paths.get('OTRO'),
            'bands': bands,
            'gsd': gsd,
            'properties': properties,
            'assets': assets
        }
        scenes.append(scene)
    return scenes

def select_best_scenes(scenes, activation_date):
    pre_scenes = []
    post_scenes = []
    for scene in scenes:
        if not scene['date']:
            continue
        if scene['date'].tzinfo is None:
            scene['date'] = scene['date'].replace(tzinfo=timezone.utc)
        if scene['date'] < activation_date:
            pre_scenes.append(scene)
        else:
            post_scenes.append(scene)
    pre_scenes.sort(key=lambda x: abs(activation_date - x['date']))
    post_scenes.sort(key=lambda x: abs(activation_date - x['date']))
    pre_scene = pre_scenes[0] if pre_scenes else None
    post_scene = post_scenes[0] if post_scenes else None
    if not pre_scene and not post_scene:
        return None, None
    return pre_scene, post_scene

def buffer_geom(building, crs):
    # Asegurar que el building tiene CRS
    if not hasattr(building, 'crs'):
        building = gpd.GeoSeries([building], crs=crs)[0]
    if crs.to_epsg() != 3857:
        building_3857 = shapely.ops.transform(
            Transformer.from_crs(crs, 3857, always_xy=True).transform, building)
        buffered = building_3857.buffer(BUFFER_METERS, cap_style=1)
        buffered = shapely.ops.transform(
            Transformer.from_crs(3857, crs, always_xy=True).transform, buffered)
    else:
        buffered = building.buffer(BUFFER_METERS, cap_style=1)
    return buffered

def rasterize_mask(poly, window, transform, out_shape):
    # Rasterización simple usando numpy y shapely
    # poly: shapely geometry
    # window: ignorado (compatibilidad)
    # transform: affine transform (x, y) -> pixel
    # out_shape: (height, width)
    mask = np.zeros(out_shape, dtype=np.uint8)
    if poly.is_empty:
        return mask
    # Obtener los puntos del polígono en píxeles
    if poly.geom_type == 'Polygon':
        polys = [poly]
    elif poly.geom_type == 'MultiPolygon':
        polys = list(poly.geoms)
    else:
        return mask
    for p in polys:
        exterior = np.array(p.exterior.coords)
        px, py = ~transform * (exterior[:, 0], exterior[:, 1])
        rr, cc = skimage.draw.polygon(py, px, shape=out_shape)
        mask[rr, cc] = 1
        for interior in p.interiors:
            interior_coords = np.array(interior.coords)
            px_i, py_i = ~transform * (interior_coords[:, 0], interior_coords[:, 1])
            rr_i, cc_i = skimage.draw.polygon(py_i, px_i, shape=out_shape)
            mask[rr_i, cc_i] = 0
    return mask

TARGET_GSD = 0.5

def sample_patches_for_percentiles(training_root, sample_size=1000):
    from random import shuffle
    sample_values = {}
    scenes = []
    for event_dir in Path(training_root).glob('*'):
        for item_json in event_dir.glob("*-calibrated/*.json"):
            try:
                with open(item_json, 'r') as f:
                    item = json.load(f)
                calib_dir = item_json.parent / "Optical_Calibration"
                if not calib_dir.exists():
                    continue
                bands = {}
                for band_file in calib_dir.glob("r-*.tif"):
                    band_name = band_file.stem.split('-')[-1]
                    bands[band_name] = band_file
                if bands:
                    scenes.append((bands, calib_dir))
            except Exception:
                continue
    shuffle(scenes)
    count = 0
    for bands, calib_dir in scenes:
        for band_name, band_path in bands.items():
            try:
                ds = gdal.Open(str(band_path))
                h, w = ds.RasterYSize, ds.RasterXSize
                if h < 64 or w < 64:
                    ds = None
                    continue
                y = np.random.randint(0, h-64)
                x = np.random.randint(0, w-64)
                patch = ds.GetRasterBand(1).ReadAsArray(x, y, 64, 64)
                ds = None
                if band_name not in sample_values:
                    sample_values[band_name] = []
                sample_values[band_name].append(patch.flatten())
                count += 1
                if count >= sample_size:
                    break
            except Exception:
                continue
        if count >= sample_size:
            break
    percentiles = {}
    for band in sample_values:
        vals = np.concatenate(sample_values[band])
        percentiles[band] = np.percentile(vals, [PERCENTILE_LOW, PERCENTILE_HIGH])
    return percentiles

# 1️⃣  al principio del archivo
BAND_ORDER = ["red","green","blue","nir","yellow","rededge","nir08","nir09","pan"]

def process_scene(scene, bbox, target_crs, is_pre, geom, fixed_mask=None, tgt_gsd=None):
    global P
    from preprocess_lib import GSD_RAW_CACHE
    if scene is None:
        logger.warning(f"process_scene: scene es None para {'PRE' if is_pre else 'POST'}")
        return None
    if not scene["bands"]:
        logger.warning(f"process_scene: scene sin bandas para {'PRE' if is_pre else 'POST'} - scene_id: {scene.get('scene_id', 'Unknown')}")
        return None

    gsd = scene.get("gsd", TARGET_GSD)
    gsd_print = gsd
    if (gsd is None or gsd == TARGET_GSD) and scene.get('scene_id') in GSD_RAW_CACHE:
        gsd_print = GSD_RAW_CACHE[scene.get('scene_id')]
        gsd = gsd_print
    if tgt_gsd is None:
        tgt_gsd = TARGET_GSD
    try:
        gsd = float(gsd)
        tgt_gsd = float(tgt_gsd)
    except Exception:
        gsd = TARGET_GSD
        tgt_gsd = TARGET_GSD

    # Solo escalar si la imagen tiene peor GSD (mayor valor)
    if abs(gsd - tgt_gsd) < 1e-6:
        scale = 1.0
    elif gsd > tgt_gsd:
        scale = gsd / tgt_gsd
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0 or scale > 100:
        logger.warning(f"Scale inválido ({scale}) para escena {scene['scene_id']} (gsd={gsd}, tgt_gsd={tgt_gsd}), usando 1.0")
        scale = 1.0
    print(f"[INFO] {scene['scene_id']} | {'PRE' if is_pre else 'POST'} | GSD original: {gsd_print} | GSD objetivo: {tgt_gsd} | scale aplicado: {scale}")
    band_arrays = []
    band_names = list(scene["bands"].keys())
    mask = fixed_mask.copy() if fixed_mask is not None else None
    for band_name, band_path in scene["bands"].items():
        try:
            ds = gdal.Open(str(band_path))
            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize
            srs = osr.SpatialReference()
            srs.ImportFromWkt(ds.GetProjection())
            raster_crs = srs.GetAttrValue('AUTHORITY', 1)
            bbox_raster = bbox
            if hasattr(target_crs, 'to_epsg') and raster_crs is not None and int(raster_crs) != target_crs.to_epsg():
                from pyproj import Transformer
                transformer = Transformer.from_crs(target_crs, int(raster_crs), always_xy=True)
                minx_arr, miny_arr = transformer.transform([bbox[0]], [bbox[1]])
                maxx_arr, maxy_arr = transformer.transform([bbox[2]], [bbox[3]])
                minx, miny = float(minx_arr[0]), float(miny_arr[0])
                maxx, maxy = float(maxx_arr[0]), float(maxy_arr[0])
                bbox_raster = (minx, miny, maxx, maxy)
            else:
                minx, miny, maxx, maxy = bbox
            inv_gt = gdal.InvGeoTransform(gt)
            if inv_gt is None or (isinstance(inv_gt, tuple) and len(inv_gt) != 6):
                logger.warning(f"No se pudo invertir el GeoTransform para banda {band_name}")
                ds = None
                continue
            px_min, py_min = gdal.ApplyGeoTransform(inv_gt, bbox_raster[0], bbox_raster[1])
            px_max, py_max = gdal.ApplyGeoTransform(inv_gt, bbox_raster[2], bbox_raster[3])
            px_min, px_max = int(np.floor(min(px_min, px_max))), int(np.ceil(max(px_min, px_max)))
            py_min, py_max = int(np.floor(min(py_min, py_max))), int(np.ceil(max(py_min, py_max)))
            if px_max <= 0 or py_max <= 0 or px_min >= w or py_min >= h:
                logger.warning(f"Ventana fuera de rango para banda {band_name} (bbox: {bbox_raster}, px: {px_min},{py_min} a {px_max},{py_max}, raster: {w}x{h})")
                ds = None
                continue
            px_min = max(0, px_min)
            py_min = max(0, py_min)
            px_max = min(w, px_max)
            py_max = min(h, py_max)
            win_w = px_max - px_min
            win_h = py_max - py_min
            if win_w <= 0 or win_h <= 0:
                logger.warning(f"Tamaño de ventana inválido para banda {band_name} (win_w: {win_w}, win_h: {win_h})")
                ds = None
                continue
            arr = ds.GetRasterBand(1).ReadAsArray(px_min, py_min, win_w, win_h)
            arr = resize(arr, (int(TILE_SIZE * scale), int(TILE_SIZE * scale)), order=1, mode='reflect', anti_aliasing=True)
            if P is not None and band_name in P:
                low, high = P[band_name]
                arr = np.clip((arr - low) / (high - low), 0, 1)
            else:
                arr = np.clip(arr, 0, 1)
            if arr.shape != (TILE_SIZE, TILE_SIZE):
                arr = resize(arr, (TILE_SIZE, TILE_SIZE), order=1, mode='reflect', anti_aliasing=True)
            band_arrays.append(arr.astype(np.float32))
            ds = None
        except Exception as e:
            logger.warning(f"Error procesando banda {band_name}: {e}")
            raise e
            continue
    print(f"[INFO] {scene['scene_id']} | {'PRE' if is_pre else 'POST'} | shape final del recorte: {band_arrays[0].shape if band_arrays else 'N/A'}")
    if not band_arrays:
        logger.warning(f"FALLO TOTAL: Todas las bandas fallaron para escena {'PRE' if is_pre else 'POST'} {scene['scene_id']}")
        return None
    if mask is None:
        logger.warning(f"No se pudo crear máscara para escena {scene['scene_id']}, usando máscara por defecto")
        mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    # === NORMALIZACIÓN POR PERCENTILES ===
    full_stack = np.stack(band_arrays, axis=-1).astype(np.float32)
    # Si más del 50% del stack es 0, devolver None
    zero_frac = np.mean(full_stack == 0)
    if zero_frac > 0.5:
        logger.info(f"Descartando escena {scene['scene_id']} por zero_frac={zero_frac:.2%}")
        return None
    # Normaliza cada banda por percentiles o min-max
    for i, band_name in enumerate(band_names):
        mn = full_stack[..., i].min()
        mx = full_stack[..., i].max()
        if P and band_name in P:
            p2, p98 = P[band_name]
            if (mx < 1 and p98 > 100) or p2 == p98:
                full_stack[..., i] = (full_stack[..., i] - mn) / max(mx - mn, 1e-8)
            else:
                full_stack[..., i] = np.clip(full_stack[..., i], p2, p98)
                full_stack[..., i] = (full_stack[..., i] - p2) / max(p98 - p2, 1e-8)
        else:
            full_stack[..., i] = (full_stack[..., i] - mn) / max(mx - mn, 1e-8)
    # --- REORDENAR A RGB SI EXISTEN ---
    desired_order = ["red", "green", "blue"]
    ordered_arrays = []
    for band in desired_order:
        if band in band_names:
            idx = band_names.index(band)
            ordered_arrays.append(full_stack[..., idx])
        else:
            ordered_arrays.append(np.zeros_like(full_stack[..., 0]))
    while len(ordered_arrays) < 3:
        ordered_arrays.append(np.zeros_like(full_stack[..., 0]))
    rgb = np.stack(ordered_arrays, axis=-1)
    # Apilar máscara después de RGB
    rgb_mask = np.concatenate([rgb, mask[..., None]], axis=-1)
    # Índices de bandas que NO son red/green/blue
    extra_idx = [i for i, b in enumerate(band_names) if b not in desired_order]
    if extra_idx:
        extra_arrays = full_stack[..., extra_idx]  # <- del stack original
        stack_final = np.concatenate([rgb_mask, extra_arrays], axis=-1)
    else:
        stack_final = rgb_mask
    try:
        return (stack_final.transpose(2, 0, 1) * 65535).round().astype(np.uint16)
    except Exception as e:
        logger.warning(f"Error en normalización final: {e}")
        return None

def align_patch(pre: np.ndarray, post: np.ndarray, max_shift: int = 8, mask=None):
    """
    Calcula el desplazamiento sub-pixel entre post y pre (RGB uint16/float32
    ya normalizados, shape (H,W,3)).
    Devuelve pre_align, (shift_y, shift_x). Si el desplazamiento estimado
    es mayor que max_shift px se devuelve el array original
    (no shift aplicado) y shift=(0,0).
    Si se pasa mask, la desplaza con order=0 (nearest).
    """
    try:
        ref = post[..., :3].mean(axis=2)
        mov = pre[..., :3].mean(axis=2)
        ref_valid = np.any(ref > 0) and np.isfinite(ref).all()
        mov_valid = np.any(mov > 0) and np.isfinite(mov).all()
        if not ref_valid or not mov_valid:
            logger.debug(f"align_patch: imagen inválida - ref_valid={ref_valid}, mov_valid={mov_valid}")
            return pre, (0.0, 0.0)
        ref_var = np.var(ref)
        mov_var = np.var(mov)
        if ref_var < 1e-6 or mov_var < 1e-6:
            logger.debug(f"align_patch: varianza insuficiente - ref_var={ref_var:.2e}, mov_var={mov_var:.2e}")
            return pre, (0.0, 0.0)
        shift_est, error, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
        dy, dx = shift_est
        if np.hypot(dy, dx) > max_shift:
            return pre, (0.0, 0.0)
        pre_aligned = ndi_shift(
            pre, shift=(dy, dx, 0), order=1, mode="nearest", cval=0
        )
        if mask is not None:
            mask_aligned = ndi_shift(mask, shift=(dy, dx), order=0, mode="nearest", cval=0)
            pre_aligned[..., 3] = mask_aligned
        return pre_aligned, (float(dy), float(dx))
    except Exception as e:
        logger.warning(f"align_patch failed: {e}")
        return pre, (0.0, 0.0)

def process_tile_worker(args):
    wkb_geom, scenes, crs_wkt, output_dir, fid, damage, idx = args
    geometry = wkb.loads(wkb_geom)
    crs = CRS.from_wkt(crs_wkt)
    return process_tile(geometry, scenes, crs, output_dir, fid, damage, idx)

def pick_scene_for_geom(scenes, geom, crs_building, is_pre, act_dt, min_cover=0.1, printed_scenes=None, fid=None):
    # 1️⃣ buffer en su CRS nativo (metros)
    buffered = buffer_geom(geom, crs_building)
    candidates = []
    if printed_scenes is None:
        printed_scenes = set()
    for scene in scenes:
        if not scene["date"]:
            continue
        if scene["date"].tzinfo is None:
            scene["date"] = scene["date"].replace(tzinfo=timezone.utc)
        is_candidate = (is_pre and scene["date"] < act_dt) or (
            not is_pre and scene["date"] >= act_dt
        )
        if not is_candidate:
            continue
        to_scene = Transformer.from_crs(crs_building, 4326, always_xy=True).transform
        buffered_wgs = sxf(to_scene, buffered)
        geom_scene = scene["geometry"]
        # Añadir validación de CRS
        if not hasattr(geom_scene, 'crs') or geom_scene.crs is None:
            geom_scene = gpd.GeoSeries([geom_scene], crs=4326)[0]
            # print(f"[FIX] Geometría de escena {scene['scene_id']} - CRS forzado a 4326")
        if not geom_scene.is_valid:
            geom_scene = geom_scene.buffer(0)
        try:
            # Debug: imprime validez de geometrías
            if is_pre:
                # print(f"[DEBUG] FID {fid} - is_valid buffer: {buffered_wgs.is_valid}, is_valid escena: {geom_scene.is_valid}")
                pass
            if geom_scene.intersects(buffered_wgs):
                intersection = geom_scene.intersection(buffered_wgs)
                cover = intersection.area / buffered_wgs.area
                if is_pre:
                    # print(f"[DEBUG] FID {fid} - Escena PRE: {scene['scene_id']} - cover={cover:.3f} - intersection empty: {intersection.is_empty}")
                    # print(f"[DEBUG] FID {fid} - CRS edificio: {crs_building}, CRS escena: {getattr(geom_scene, 'crs', 'N/A')}")
                    pass
            else:
                cover = 0.0
                if is_pre:
                    # print(f"[DEBUG] FID {fid} - Escena PRE: {scene['scene_id']} - NO INTERSECTA")
                    pass
            # Solo imprime si no se ha impreso para este scene_id
            if scene['scene_id'] not in printed_scenes:
                # print(f"{scene['scene_id']}   cover={cover:.3f}")
                printed_scenes.add(scene['scene_id'])
            # 4️⃣ fallback: si cover==0 y el centroide cae dentro, pon cover=1e-3
            if cover == 0 and geom_scene.contains(buffered_wgs.centroid):
                cover = 1e-3
            # [CHECK] print para ver si se añade el candidato
            # print(f"[CHECK] FID {fid} - cover={cover}, centroid dentro: {geom_scene.contains(buffered_wgs.centroid)}")
        except Exception as e:
            logger.warning(
                f"Error calculando intersección para escena {scene['scene_id']}: {e}"
            )
            continue
        if cover > 0 or geom_scene.contains(buffered_wgs.centroid):
            candidates.append((scene, cover))
    if not candidates:
        return None
    candidates.sort(
        key=lambda t: (
            t[1] < 0.95,
            -t[1],
            abs(act_dt - t[0]["date"]).total_seconds(),
            t[0].get("cloud_cover", 999),
            t[0].get("sat:off_nadir", 999)
        )
    )
    if len(candidates) > 1:
        pass
    return candidates[0][0]

def process_tile(building, scenes, crs, output_dir, fid, damage_label, idx):
    # scenes ahora es (pre_scene, post_scene, activation_date)
    pre_scene, post_scene, activation_date = scenes
    if pre_scene is None and post_scene is None:
        logger.warning(f"FID {fid}: sin PRE ni POST; descarto")
        return None
    if pre_scene is None:
        logger.warning(f"FID {fid}: sin PRE; relleno con ceros")
    if post_scene is None:
        logger.warning(f"FID {fid}: sin POST; relleno con ceros")
    buffered_geom = buffer_geom(building, crs)
    if not buffered_geom.is_valid:
        buffered_geom = buffered_geom.buffer(0)
    mask = np.zeros((224, 224), dtype=np.uint8)
    try:
        from rasterio.transform import from_bounds
        import rasterio
        import shapely
        from rasterio.features import rasterize
        bbox = buffered_geom.bounds
        transform = from_bounds(*bbox, 224, 224)
        mask = rasterize(
            [(shapely.geometry.mapping(buffered_geom), 1)],
            out_shape=(224, 224),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
    except Exception as e:
        logger.warning(f"No se pudo rasterizar la máscara final para FID {fid}: {e}")
        mask = np.ones((224, 224), dtype=np.uint8)
    bbox = buffered_geom.bounds
    pre_data = (
        process_scene(pre_scene, bbox, crs, is_pre=True, geom=building, fixed_mask=mask)
        if pre_scene
        else np.zeros((len(BAND_ORDER)+1, TILE_SIZE, TILE_SIZE), np.uint16)
    )
    post_data = (
        process_scene(post_scene, bbox, crs, is_pre=False, geom=building, fixed_mask=mask)
        if post_scene
        else np.zeros((len(BAND_ORDER)+1, TILE_SIZE, TILE_SIZE), np.uint16)
    )
    shift_xy = (0.0, 0.0)
    if pre_data is not None and post_data is not None and pre_scene is not None and post_scene is not None:
        pre_data_hwc = pre_data.transpose(1, 2, 0)
        post_data_hwc = post_data.transpose(1, 2, 0)
        aligned_pre_hwc, shift_xy = align_patch(pre_data_hwc, post_data_hwc, mask=mask)
        pre_data = aligned_pre_hwc.transpose(2, 0, 1)
        logger.debug(f"fid={fid}  shift={shift_xy}")
    has_pre = int(pre_scene is not None)
    has_post = int(post_scene is not None)
    if damage_label is None or (isinstance(damage_label, float) and math.isnan(damage_label)):
        damage_label = -1
    # Guardar el archivo con nombre simplificado si fid==idx
    tile_path = output_dir / f"{str(fid)}.npz"
    np.savez_compressed(tile_path, pre=pre_data, post=post_data)
    return dict(
        fid=fid,
        tile_path=str(tile_path),
        damaged=int(damage_label),
        has_pre=has_pre,
        has_post=has_post,
        shift_x=shift_xy[1],
        shift_y=shift_xy[0],
    )

def find_fid_column_with_fiona(gpkg_path, gdf):
    with fiona.open(gpkg_path) as src:
        schema_columns = src.schema['properties'].keys()
        # print(f"Columnas en esquema Fiona: {list(schema_columns)}")
        if 'id' in schema_columns and 'id' not in gdf.columns:
            gdf['id'] = gdf.index
            logger.info("La columna 'id' detectada por Fiona se ha asignado al índice como 'id'.")
        elif 'fid' in schema_columns and 'fid' not in gdf.columns:
            gdf['id'] = gdf.index
            logger.info("La columna 'fid' detectada por Fiona se ha asignado al índice como 'id'.")
    return gdf 

def process_gpkg(gpkg_path, training_root, output_root, split):
    logger.info(f"Procesando {gpkg_path}")
    city_area = gpkg_path.stem.replace('_Training', '').replace('_training', '').replace('_Testing', '').replace('_testing', '').lower()
    output_dir = output_root / split / city_area
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = load_gpkg_with_fiona(gpkg_path)
    logger.info(f"Edificios en GPKG: {len(gdf)}")
    if "afghanistan" in city_area:
        logger.info(f"Columnas en {gpkg_path}: {list(gdf.columns)}")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    gdf['geometry'] = gdf['geometry'].buffer(0)
    logger.info(f"Edificios tras limpieza: {len(gdf)}")
    if gdf.crs is None:
        logger.warning(f"GPKG {gpkg_path} no tiene CRS. Asignando EPSG:4326 por defecto.")
        gdf.set_crs(4326, inplace=True)
    aoi_geom_wgs = gdf.to_crs(4326).geometry.union_all()
    fid_col = find_fid_column(gdf)
    if not fid_col:
        gdf = find_fid_column_with_fiona(gpkg_path, gdf)
        fid_col = find_fid_column(gdf)
        if not fid_col:
            logger.warning(
                f"No se encontró columna id/fid en {gpkg_path} ni con fiona, usando índice como id."
            )
            gdf["id"] = gdf.index
            fid_col = "id"
    logger.info(f"Columna FID: {fid_col}")
    event_name = gpkg_path.parent.name
    call_dir = training_root / event_name
    activation_date = get_activation_date(call_dir)
    logger.info(f"Fecha de activación (terremoto): {activation_date}")
    interval_start, interval_end = None, None
    coll_json = call_dir / "collection.json"
    if coll_json.exists():
        with open(coll_json, "r", encoding="utf-8") as f:
            coll = json.load(f)
        try:
            interval = coll['extent']['temporal']['interval'][0]
            interval_start = datetime.fromisoformat(interval[0].replace('Z', '+00:00'))
            interval_end = datetime.fromisoformat(interval[1].replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"Error leyendo intervalo temporal en {coll_json}: {e}")
    else:
        logger.warning(f"No se encontró collection.json en {call_dir}")
    scenes = find_scenes_for_aoi(aoi_geom_wgs, call_dir, interval_start, interval_end)
    logger.info(f"Escenas encontradas: {len(scenes)}")
    if not scenes:
        logger.warning(f"No se encontraron escenas para {gpkg_path}")
        return []
    pre_scene_global, post_scene_global = select_best_scenes(scenes, activation_date)
    logger.info(f"Fecha mejor escena PRE: {pre_scene_global['date'] if pre_scene_global else 'N/A'}")
    logger.info(f"Fecha mejor escena POST: {post_scene_global['date'] if post_scene_global else 'N/A'}")
    meta_data = []
    damage_col = 'damaged' if 'damaged' in gdf.columns else None
    logger.info(f"Columna de daño: {damage_col}")
    crs_wkt = gdf.crs.to_wkt()
    # Precalcula la mejor escena PRE/POST para cada tile
    pre_post_for_fid = {}
    for idx, row in gdf.iterrows():
        fid = row[fid_col]
        geom = row['geometry']
        pre_scene = pick_scene_for_geom(scenes, geom, gdf.crs, True, activation_date)
        post_scene = pick_scene_for_geom(scenes, geom, gdf.crs, False, activation_date)
        pre_post_for_fid[fid] = (pre_scene, post_scene)
    tasks = []
    for idx, row in gdf.iterrows():
        fid = row[fid_col]
        damage = row[damage_col] if damage_col else -1
        pre_scene, post_scene = pre_post_for_fid[fid]
        # Pasar solo las mejores escenas a process_tile_worker
        task = (row.geometry.wkb, (pre_scene, post_scene, activation_date), crs_wkt, output_dir, fid, damage, idx)
        tasks.append(task)
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    executor = None
    def signal_handler(signum, frame):
        if executor is not None:
            logger.info("\nCancelando procesos...")
            executor.shutdown(wait=False, cancel_futures=True)
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        sys.exit(1)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        with ProcessPoolExecutor(max_workers=12) as executor:
            try:
                for meta in tqdm(executor.map(process_tile_worker, tasks, chunksize=8192), 
                               total=len(tasks), 
                               desc=f"Procesando {gpkg_path.name}"):
                    if meta:
                        meta['gpkg'] = gpkg_path.name
                        meta['event'] = event_name
                        meta['split'] = split
                        meta_data.append(meta)
            except KeyboardInterrupt:
                logger.info("\nInterrupción detectada. Limpiando...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
    logger.info(f"Tiles generados para {gpkg_path}: {len(meta_data)}")
    return meta_data 

# Exportar BAND_ORDER y pick_scene_for_geom
__all__ = [
    'BAND_ORDER',
    'pick_scene_for_geom',
    'load_gpkg_with_fiona',
    'find_scenes_for_aoi',
    'buffer_geom',
    'process_scene',
    'align_patch',
    'get_activation_date'
] 