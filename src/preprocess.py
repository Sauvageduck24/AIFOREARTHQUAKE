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
import rasterio
import shapely
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.windows import from_bounds
from scipy.ndimage import shift as ndi_shift
from shapely import wkb
from shapely.geometry import box, mapping, shape
from skimage.registration import phase_cross_correlation
import json
from skimage.transform import resize
from tqdm import tqdm
import argparse
import fiona

# Configuración global
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", 'GeoSeries.notna', UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
ACTIVATION_DATES = {
    "Call_919_Turkey": datetime(2023, 2, 6, tzinfo=timezone.utc),
    "Call_1095_Myanmar": datetime(2025, 3, 23, tzinfo=timezone.utc),
    "Call_1075_China": datetime(2025, 1, 8, tzinfo=timezone.utc),
    "Call_920_Syria": datetime(2023, 2, 9, tzinfo=timezone.utc),
    "Call_965_Morocco": datetime(2023, 9, 10, tzinfo=timezone.utc),
    "Call_970_Afghanistan": datetime(2023, 10, 9, tzinfo=timezone.utc)
}

TILE_SIZE = 224

BUFFER_METERS = 200  # Buffer consistente en metros
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

def find_fid_column(gdf):
    # Primero buscar columnas que empiecen con 'fid_'
    fid_cols = [col for col in gdf.columns if col.lower().startswith('fid_')]
    if fid_cols:
        return fid_cols[0]
    # Si no hay columnas fid_*, buscar columna 'id'
    if 'id' in gdf.columns:
        return 'id'
    # Si no hay ninguna, devolver None
    return None

def load_gpkg_with_fiona(gpkg_path):
    """Carga un archivo GPKG usando Fiona y lo convierte a GeoDataFrame."""
    features = []
    feature_ids = []
    selected_fid_column = None  # Guardar qué columna usamos
    crs = None
    
    with fiona.open(gpkg_path) as src:
        crs = src.crs
        schema_columns = list(src.schema['properties'].keys())
        logger.debug(f"Columnas disponibles en el esquema: {schema_columns}")
        
        # Buscar columnas que empiecen con 'fid_'
        fid_columns = [col for col in schema_columns if col.lower().startswith('fid_')]
        
        if fid_columns:
            # Si encontramos columnas fid_*, usar la primera como identificador
            selected_fid_column = fid_columns[0]
            logger.info(f"Encontrada columna FID específica: {selected_fid_column}")
            
            for feature in src:
                features.append(feature)
                # Usar el valor de la columna fid_* como identificador
                fid_value = feature['properties'].get(selected_fid_column)
                feature_ids.append(fid_value)
        else:
            # Si no hay columnas fid_*, usar el id del feature de Fiona
            logger.info("No se encontraron columnas fid_*, usando ID de Fiona")
            
            for feature in src:
                features.append(feature)
                # Extraer el id del feature de Fiona
                feature_ids.append(feature.get('id'))
    
    # Convertir a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    
    # Asignar los IDs a la columna apropiada
    if selected_fid_column:
        # Si usamos una columna fid_*, mantener ese nombre
        gdf[selected_fid_column] = feature_ids
        logger.info(f"Asignados {len(feature_ids)} IDs a la columna '{selected_fid_column}'")
    else:
        # Si usamos el ID de Fiona, asignar a columna 'id'
        gdf['id'] = feature_ids
        logger.info(f"Asignados {len(feature_ids)} IDs a la columna 'id'")
    
    return gdf

def extract_date_from_name(name):
    m = re.search(r"(\d{8})", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    return None

def get_activation_date(call_dir: Path) -> datetime | None:
    # 1️⃣ Intenta leer collection.json
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
                pass    # formato raro → cae al fallback

    # 2️⃣ Busca en diccionario hard-code
    dt = ACTIVATION_DATES.get(call_dir.name)
    if dt:
        return dt

    # 3️⃣ Sin fecha conocida
    logger.warning(f"No activation date for {call_dir.name}")
    return None

def read_gsd(item_json):
    """Lee el Ground Sample Distance (GSD) del item JSON o lo estima del primer band."""
    with open(item_json) as f:
        item = json.load(f)
    gsd = item['properties'].get('gsd')
    
    if gsd is None:
        # Estimate from the first band
        calib_dir = item_json.parent / "Optical_Calibration"
        band_files = list(calib_dir.glob('r-*.tif'))
        if band_files:
            with rasterio.open(band_files[0]) as src:
                gsd = abs(src.transform.a)
    else:
        # Validar que el GSD del JSON sea razonable
        if gsd < 0.1 or gsd > 100:  # GSD fuera del rango razonable para imágenes satelitales
            #print(f"⚠️  GSD del JSON sospechoso: {gsd} para {item_json.parent.name}")
            #print(f"   Estimando GSD desde el raster...")
            # Estimar desde el primer band como fallback
            calib_dir = item_json.parent / "Optical_Calibration"
            band_files = list(calib_dir.glob('r-*.tif'))
            if band_files:
                with rasterio.open(band_files[0]) as src:
                    gsd_from_raster = abs(src.transform.a)
                    #print(f"   GSD del raster: {gsd_from_raster}")
                    if 0.1 <= gsd_from_raster <= 100:  # Si el GSD del raster es razonable
                        gsd = gsd_from_raster
                        #print(f"   ✅ Usando GSD del raster: {gsd}")
                    else:
                        #print(f"   ❌ GSD del raster también sospechoso, usando valor por defecto")
                        gsd = None  # Se manejará en process_scene
    
    return gsd

def find_scenes_for_aoi(aoi_geom, training_root, interval_start=None, interval_end=None):
    scenes = []
    training_root = Path(training_root)
    for item_json in training_root.glob("*-calibrated/*.json"):
        try:
            with open(item_json, 'r') as f:
                item = json.load(f)
            if 'geometry' in item:
                scene_geom = shape(item['geometry'])
            elif 'bbox' in item:
                scene_geom = box(*item['bbox'])
            else:
                continue
            if not aoi_geom.intersects(scene_geom):
                continue
            calib_dir = item_json.parent / "Optical_Calibration"
            if not calib_dir.exists():
                continue
            bands = {}
            for band_file in calib_dir.glob("r-*.tif"):
                band_name = band_file.stem.split('-')[-1]
                bands[band_name] = band_file
            # Añadir pan si existe
            pan_file = calib_dir / "r-pan.tif"
            if pan_file.exists():
                bands['pan'] = pan_file
            scene_date = item.get('properties', {}).get('datetime')
            if scene_date:
                try:
                    scene_date = datetime.fromisoformat(scene_date.replace('Z', '+00:00'))
                except:
                    scene_date = extract_date_from_name(item_json.stem)
            # Filter by temporal interval
            if interval_start and interval_end and scene_date:
                # Aplicar filtro de inicio estricto
                if scene_date < interval_start:
                    continue
                
                # Aplicar filtro de fin con margen configurable
                if scene_date > interval_end + timedelta(hours=MAX_EXTENT_OVERLAP_HRS):
                    logger.warning(f"{item_json.parent.name} fuera de intervalo "
                                 f"({scene_date} > {interval_end})  — descartado")
                    continue
                elif scene_date > interval_end:
                    hours_over = (scene_date - interval_end).total_seconds() / 3600
                    logger.info(f"{item_json.parent.name} {scene_date} a {hours_over:.1f} h "
                               f"del límite; lo acepto por margen +{MAX_EXTENT_OVERLAP_HRS} h")
            # Read GSD
            gsd = read_gsd(item_json)
            if gsd:
                logger.debug(f"GSD para escena {item_json.parent.name}: {gsd} m/px")
            
            properties = item.get('properties', {})
            scene = {
                'scene_id': item_json.parent.name,
                'date': scene_date,
                'geometry': scene_geom,
                'path': item_json.parent,
                'bands': bands,
                'gsd': gsd,
                'cloud_cover': properties.get('eo:cloud_cover'),
                'sat:off_nadir': properties.get('sat:off_nadir')
            }
            logger.debug(f"Scene structure: {scene}")
            # Validate scene structure before appending
            if isinstance(scene, dict) and 'date' in scene and 'geometry' in scene:
                scenes.append(scene)
            else:
                logger.warning(f"Invalid scene structure: {scene}")
        except Exception as e:
            logger.warning(f"Error procesando {item_json}: {e}")
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
    # Buffer consistente en metros
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
    """Devuelve una máscara uint8 (1=dentro, 0=fuera) del polígono en
    coordenadas de imagen ya recortada."""
    mask = rasterize(
        [(mapping(poly), 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask


TARGET_GSD = 0.5  # metros por pixel objetivo


def sample_patches_for_percentiles(training_root, sample_size=1000):
    """Extrae parches aleatorios de todas las bandas y sensores para calcular percentiles globales."""
    from random import shuffle
    sample_values = {}
    scenes = []
    # Buscar todas las escenas
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
                with rasterio.open(band_path) as src:
                    # Extraer un parche aleatorio
                    h, w = src.height, src.width
                    if h < 64 or w < 64:
                        continue
                    y = np.random.randint(0, h-64)
                    x = np.random.randint(0, w-64)
                    patch = src.read(1, window=rasterio.windows.Window(x, y, 64, 64))
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
    # Concatenar y calcular percentiles
    percentiles = {}
    for band in sample_values:
        vals = np.concatenate(sample_values[band])
        percentiles[band] = np.percentile(vals, [PERCENTILE_LOW, PERCENTILE_HIGH])
    return percentiles


def process_scene(scene, bbox, target_crs, is_pre, geom):
    global P
    if scene is None:
        logger.warning(f"process_scene: scene es None para {'PRE' if is_pre else 'POST'}")
        return None
    if not scene["bands"]:
        logger.warning(f"process_scene: scene sin bandas para {'PRE' if is_pre else 'POST'} - scene_id: {scene.get('scene_id', 'Unknown')}")
        return None
    
    gsd = scene.get("gsd", TARGET_GSD)
    
    # Robustez: gsd y TARGET_GSD deben ser finitos y positivos
    try:
        gsd = float(gsd)
    except Exception:
        #print(f"❌ GSD no convertible a float: {gsd} en escena {scene['scene_id']}. Usando {TARGET_GSD}")
        gsd = TARGET_GSD
    
    if not np.isfinite(gsd) or gsd <= 0:
        #print(f"❌ GSD inválido ({gsd}) para escena {scene['scene_id']}, usando valor por defecto {TARGET_GSD}")
        gsd = TARGET_GSD
    
    # Detectar GSD extremadamente pequeño o grande que causa problemas
    if gsd < 0.1:
        #print(f"⚠️  GSD demasiado pequeño: {gsd} para escena {scene['scene_id']}")
        #print(f"   Probable error en metadatos. Forzando a TARGET_GSD = {TARGET_GSD}")
        gsd = TARGET_GSD
    elif gsd > 100:
        #print(f"⚠️  GSD demasiado grande: {gsd} para escena {scene['scene_id']}")
        #print(f"   Probable error en metadatos. Forzando a TARGET_GSD = {TARGET_GSD}")
        gsd = TARGET_GSD
    
    tgt_gsd = TARGET_GSD
    try:
        tgt_gsd = float(tgt_gsd)
    except Exception:
        logger.warning(f"TARGET_GSD no convertible a float: {tgt_gsd}. Usando 1.0")
        tgt_gsd = 1.0
    if not np.isfinite(tgt_gsd) or tgt_gsd <= 0:
        logger.warning(f"TARGET_GSD inválido ({tgt_gsd}), usando 1.0")
        tgt_gsd = 1.0
    scale = gsd / tgt_gsd
    if not np.isfinite(scale) or scale <= 0:
        logger.warning(f"Scale inválido ({scale}) para escena {scene['scene_id']} (gsd={gsd}, target_gsd={tgt_gsd}), usando 1.0")
        scale = 1.0
    if NO_UPSAMPLE and scale > 1.0:
        scale = 1.0  # no ampliamos escenas de baja resolución
    
    # Validación adicional para scale muy pequeño que podría causar dimensiones 0
    if scale < 0.001:  # Si el scale es menor a 0.001, es prácticamente 0
        #print(f"⚠️  Scale extremadamente pequeño detectado: {scale:.10f}")
        #print(f"   Forzando scale = 1.0 para evitar dimensiones inválidas")
        scale = 1.0
    
    band_arrays = []
    band_names = list(scene["bands"].keys())
    mask = None
    
    for band_name, band_path in scene["bands"].items():
        try:
            with rasterio.open(band_path) as src:
                if src.crs != target_crs:
                    tf = Transformer.from_crs(
                        target_crs, src.crs, always_xy=True
                    ).transform
                    minx, miny = tf(bbox[0], bbox[1])
                    maxx, maxy = tf(bbox[2], bbox[3])
                    bbox_t = (minx, miny, maxx, maxy)
                else:
                    bbox_t = bbox
                
                win = from_bounds(*bbox_t, transform=src.transform)
                
                if win.width == 0 or win.height == 0:
                    logger.warning(f"Ventana vacía para escena {scene['scene_id']}, banda {band_name}: {win}")
                    continue
                
                data = src.read(1, window=win, boundless=True, fill_value=0)
                if data.size == 0:
                    print(f"❌ Datos vacíos para banda {band_name} en escena {scene['scene_id']}")
                    continue

                if mask is None:
                    # Asegurarse de que el polígono está en el mismo CRS que el raster
                    if src.crs != target_crs:
                        geom_ds = shapely.ops.transform(
                            Transformer.from_crs(
                                target_crs, src.crs, always_xy=True
                            ).transform,
                            geom,
                        )
                    else:
                        geom_ds = geom

                    # Crear máscara en resolución nativa
                    mask_full = rasterize_mask(
                        poly=geom_ds,  # shapely Polygon del edificio
                        window=win,
                        transform=src.window_transform(win),
                        out_shape=(int(win.height), int(win.width)),
                    )

                    # Validar que mask_full no esté vacía
                    if mask_full.size == 0 or mask_full.shape[0] == 0 or mask_full.shape[1] == 0:
                        print(f"❌ Máscara vacía para banda {band_name} en escena {scene['scene_id']}")
                        continue

                    # Validar que data tenga dimensiones válidas antes de calcular target_h/target_w
                    if data.shape[0] == 0 or data.shape[1] == 0:
                        print(f"❌ Datos con dimensiones inválidas {data.shape} para banda {band_name} en escena {scene['scene_id']}")
                        continue

                    # Re-muestrear la máscara con el MISMO pipeline que el RGB
                    target_h = int(data.shape[0] * scale)
                    target_w = int(data.shape[1] * scale)
                    
                    # Validar que las dimensiones objetivo sean válidas
                    if target_h <= 0 or target_w <= 0:
                        print(f"❌ Dimensiones objetivo inválidas {target_h}x{target_w} para banda {band_name} en escena {scene['scene_id']}")
                        continue

                    if scale != 1.0:
                        # Validar dimensiones antes de resize
                        if target_h <= 0 or target_w <= 0:
                            print(f"❌ Dimensiones inválidas para resize: {target_h}x{target_w}")
                            continue
                        mask_full = resize(
                            mask_full,
                            (target_h, target_w),
                            order=0,
                            mode="edge",
                            preserve_range=True,
                            anti_aliasing=False,
                        )

                    # Validar dimensiones antes de cv2.resize
                    if mask_full.size == 0 or mask_full.shape[0] == 0 or mask_full.shape[1] == 0:
                        print(f"❌ Máscara vacía después de resize para banda {band_name}")
                        continue

                    mask = cv2.resize(
                        mask_full.astype(np.uint8),
                        (TILE_SIZE, TILE_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)

                # Validar que data no esté vacía
                if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
                    print(f"❌ Datos vacíos para banda {band_name} antes del resize")
                    continue
                
                # --- Remuestreo según escala ---
                if scale == 1.0:
                    data = cv2.resize(
                        data.astype(np.float32),
                        (TILE_SIZE, TILE_SIZE),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    # Control extra: scale debe ser finito y positivo
                    if not np.isfinite(scale) or scale <= 0:
                        logger.error(f"Scale inválido ({scale}) justo antes de resize para escena {scene['scene_id']}, banda {band_name}. Rellenando con ceros.")
                        data = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    else:
                        target_h = int(TILE_SIZE * scale)
                        target_w = int(TILE_SIZE * scale)
                        if target_h <= 0 or target_w <= 0 or not np.isfinite(target_h) or not np.isfinite(target_w):
                            logger.error(f"target_h/w inválidos ({target_h}, {target_w}) para escena {scene['scene_id']}, banda {band_name}. Rellenando con ceros.")
                            data = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                        else:
                            data = resize(
                                data,
                                (target_h, target_w),
                                order=3,
                                mode="edge",
                                preserve_range=True,
                                anti_aliasing=True,
                            )
                            if data.shape != (TILE_SIZE, TILE_SIZE):
                                data = cv2.resize(
                                    data.astype(np.float32),
                                    (TILE_SIZE, TILE_SIZE),
                                    interpolation=cv2.INTER_AREA
                                )
                band_arrays.append(data)
        except Exception as e:
            print(f"❌ Error procesando banda {band_name}: {e}")
            continue
    if not band_arrays:
        print(f"❌ FALLO TOTAL: Todas las bandas fallaron para escena {'PRE' if is_pre else 'POST'} {scene['scene_id']}")
        return None
    
    # Si no se creó máscara (por problemas con geometrías), crear una máscara por defecto
    if mask is None:
        print(f"⚠️  No se pudo crear máscara para escena {scene['scene_id']}, usando máscara por defecto")
        mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    
    # === NORMALIZACIÓN POR PERCENTILES ===
    full_stack = np.stack(band_arrays, axis=-1).astype(np.float32)
    # Si más del 50% del stack es 0, devolver None
    zero_frac = np.mean(full_stack == 0)
    if zero_frac > 0.5:
        #print(f"❌ Más del 50% del stack es 0 (frac={zero_frac:.2f}), tile descartado para escena {scene['scene_id']}")
        return None
    for i, band_name in enumerate(band_names):
        if P and band_name in P:
            p2, p98 = P[band_name]
            full_stack[..., i] = np.clip(full_stack[..., i], p2, p98)
            full_stack[..., i] = (full_stack[..., i] - p2) / max(p98 - p2, 1e-8)
        else:
            mn = full_stack[..., i].min()
            mx = full_stack[..., i].max()
            full_stack[..., i] = (full_stack[..., i] - mn) / max(mx - mn, 1e-8)

    # --- REORDENAR A RGB SI EXISTEN ---
    desired_order = ["red", "green", "blue"]
    ordered_arrays = []
    for band in desired_order:
        if band in band_names:
            idx = band_names.index(band)
            ordered_arrays.append(full_stack[..., idx])
        else:
            # Si falta la banda, rellena con ceros
            ordered_arrays.append(np.zeros_like(full_stack[..., 0]))
    # Si hay menos de 3 bandas, rellena hasta 3
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

    #print(f"✅ Escena {scene['scene_id']} procesada exitosamente - shape final: {stack_final.shape}")
    try:
        return (stack_final.transpose(2, 0, 1) * 65535).round().astype(np.uint16)
    except Exception as e:
        logger.warning(f"Error en normalización final: {e}")
        return None


def align_patch(pre: np.ndarray, post: np.ndarray, max_shift: int = 8):
    """
    Calcula el desplazamiento sub-pixel entre post y pre (RGB uint16/float32
    ya normalizados, shape (H,W,3)).
    Devuelve pre_align, (shift_y, shift_x). Si el desplazamiento estimado
    es mayor que max_shift px se devuelve el array original
    (no shift aplicado) y shift=(0,0).
    """
    try:
        # skimage mira canales (H,W) → pasamos luminancia aproximada
        ref = post[..., :3].mean(axis=2)
        mov = pre[..., :3].mean(axis=2)
        
        # Validar que las imágenes no estén vacías
        ref_valid = np.any(ref > 0) and np.isfinite(ref).all()
        mov_valid = np.any(mov > 0) and np.isfinite(mov).all()
        
        if not ref_valid or not mov_valid:
            logger.debug(f"align_patch: imagen inválida - ref_valid={ref_valid}, mov_valid={mov_valid}")
            return pre, (0.0, 0.0)
        
        # Verificar que las imágenes tienen varianza suficiente
        ref_var = np.var(ref)
        mov_var = np.var(mov)
        if ref_var < 1e-6 or mov_var < 1e-6:
            logger.debug(f"align_patch: varianza insuficiente - ref_var={ref_var:.2e}, mov_var={mov_var:.2e}")
            return pre, (0.0, 0.0)
        
        shift_est, error, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
        dy, dx = shift_est  # (row, col)
        if np.hypot(dy, dx) > max_shift:
            return pre, (0.0, 0.0)
        pre_aligned = ndi_shift(
            pre, shift=(dy, dx, 0), order=1, mode="nearest", cval=0
        )
        return pre_aligned, (float(dy), float(dx))
    except Exception as e:
        logger.warning(f"align_patch failed: {e}")
        return pre, (0.0, 0.0)


def process_tile_worker(args):
    wkb_geom, scenes, crs_wkt, output_dir, fid, damage = args
    geometry = wkb.loads(wkb_geom)
    crs = CRS.from_wkt(crs_wkt)
    return process_tile(geometry, scenes, crs, output_dir, fid, damage)


def pick_scene_for_geom(scenes, geom, crs, is_pre, act_dt, min_cover=0.8):
    """
    Devuelve la escena que contiene el edificio con una cobertura mínima y cuya fecha está más cercana al activation_date
    (antes para pre, después/igual para post). Si no hay ninguna, devuelve None.
    En caso de empate en cobertura, desempata por cercanía temporal.
    """
    buffered_geom = buffer_geom(geom, crs)
    candidates = []
    for scene in scenes:
        if not scene["date"]:
            continue

        # Asegurarse que la fecha de la escena tiene timezone
        if scene["date"].tzinfo is None:
            scene["date"] = scene["date"].replace(tzinfo=timezone.utc)

        # Comprobar si la escena es pre o post
        is_candidate = (is_pre and scene["date"] < act_dt) or (
            not is_pre and scene["date"] >= act_dt
        )
        if not is_candidate:
            continue

        # 3. Umbral temporal para escenas pre-evento
        if is_pre and (act_dt - scene["date"]).days > 550:
            continue

        # Calcular cobertura
        try:
            intersection = scene["geometry"].intersection(buffered_geom)
            cover = intersection.area / buffered_geom.area
        except Exception as e:
            logger.warning(
                f"Error calculando intersección para escena {scene['scene_id']}: {e}"
            )
            cover = 0

        if cover >= min_cover:
            candidates.append((scene, cover))

    if not candidates:
        # logger.debug(
        #     f"No scene con cobertura >={min_cover*100}% para {'pre' if is_pre else 'post'} en fecha {act_dt}"
        # )
        return None

    # Ordenar priorizando cobertura alta (>95%) y luego cercanía temporal
    # Si varias escenas tienen cobertura similar (diferencia <5%), priorizar cercanía temporal
    def sort_key(t):
        scene, cover = t
        temporal_dist = abs(act_dt - scene["date"]).total_seconds()
        cloud_cover = scene.get("cloud_cover", 999)
        off_nadir = scene.get("sat:off_nadir", 999)
        
        # Si la cobertura es casi perfecta (>0.95), priorizar tiempo
        if cover > 0.95:
            return (0, temporal_dist, cloud_cover, off_nadir)
        else:
            # Para coberturas menores, priorizar cobertura primero
            return (1, -cover, temporal_dist, cloud_cover, off_nadir)
    
    candidates.sort(key=sort_key)

    # Debug: mostrar los primeros candidatos para entender la selección
    if len(candidates) > 1:
        logger.debug(f"Scene selection for {'PRE' if is_pre else 'POST'}:")
        for i, (scene, cover) in enumerate(candidates[:3]):  # Solo los 3 primeros
            time_diff = abs(act_dt - scene["date"]).total_seconds() / 3600
            logger.debug(f"  {i+1}. {scene['scene_id']} - Cover: {cover:.3f}, Time: {time_diff:.1f}h")

    # Loguear la escena seleccionada y sus competidoras
    # best_scene, best_cover = candidates[0]
    # logger.info(
    #     f"Seleccionada {'PRE' if is_pre else 'POST'}: {best_scene['scene_id']} "
    #     f"({best_scene['date'].date()}) con cover {best_cover:.2f}. "
    #     f"Competidoras: {len(candidates)-1}"
    # )
    return candidates[0][0]


def process_tile(building, scenes, crs, output_dir, fid, damage_label):
    all_scenes, activation_date = scenes
    # logger.debug(f"All scenes before select_best_scenes: {all_scenes}")
    pre_scene_global, post_scene_global = select_best_scenes(all_scenes, activation_date)
    pre_scene = (
        pick_scene_for_geom(all_scenes, building, crs, True, activation_date)
        or pre_scene_global
    )
    post_scene = (
        pick_scene_for_geom(all_scenes, building, crs, False, activation_date)
        or post_scene_global
    )

    # Validación extra de fechas pre/post
    if pre_scene and post_scene:
        def _closest(scene_list, ref_date):
            return min(scene_list, key=lambda s: abs(s["date"] - ref_date))

        # 1.1 chequeo "¿hay otra PRE más reciente?" - solo avisar si hay diferencia significativa
        pre_candidates = [s for s in all_scenes if s["date"] < activation_date]
        if pre_candidates:
            closer_pre = _closest(pre_candidates, activation_date)
            if closer_pre["scene_id"] != pre_scene["scene_id"]:
                # Solo mostrar warning si la diferencia temporal es significativa (>6 horas)
                time_diff_hours = abs(closer_pre["date"] - pre_scene["date"]).total_seconds() / 3600
                if time_diff_hours > 6:
                    logger.warning(
                        f"Hay un PRE más cercano: {closer_pre['scene_id']} "
                        f"{closer_pre['date']}  (usando {pre_scene['scene_id']}) - diff: {time_diff_hours:.1f}h"
                    )

        # 1.2 idem para POST
        post_candidates = [s for s in all_scenes if s["date"] >= activation_date]
        if post_candidates:
            closer_post = _closest(post_candidates, activation_date)
            if closer_post["scene_id"] != post_scene["scene_id"]:
                # Solo mostrar warning si la diferencia temporal es significativa (>6 horas)
                time_diff_hours = abs(closer_post["date"] - post_scene["date"]).total_seconds() / 3600
                if time_diff_hours > 6:
                    logger.warning(
                        f"Hay un POST más cercano: {closer_post['scene_id']} "
                        f"{closer_post['date']}  (usando {post_scene['scene_id']}) - diff: {time_diff_hours:.1f}h"
                    )

        # Validación temporal
        if not (pre_scene["date"] < activation_date <= post_scene["date"]):
            logger.error("¡Incoherencia temporal! Abort tile.")
            return None

    # Buffer consistente
    buffered_geom = buffer_geom(building, crs)
    # Sanity-check para evitar geometrías self-intersecting
    if not buffered_geom.is_valid:
        buffered_geom = buffered_geom.buffer(0)
    bbox = buffered_geom.bounds
    pre_data = (
        process_scene(pre_scene, bbox, crs, is_pre=True, geom=building)
        if pre_scene
        else None
    )
    post_data = (
        process_scene(post_scene, bbox, crs, is_pre=False, geom=building)
        if post_scene
        else None
    )

    shift_xy = (0.0, 0.0)
    if pre_data is not None and post_data is not None:
        # Transpose from (C, H, W) to (H, W, C) for alignment
        pre_data_hwc = pre_data.transpose(1, 2, 0)
        post_data_hwc = post_data.transpose(1, 2, 0)

        aligned_pre_hwc, shift_xy = align_patch(pre_data_hwc, post_data_hwc)

        # Transpose back to (C, H, W) for saving
        pre_data = aligned_pre_hwc.transpose(2, 0, 1)
        logger.debug(f"fid={fid}  shift={shift_xy}")

    if pre_data is None:
        #logger.warning(f"FID {fid}: PRE data es None - escena seleccionada: {pre_scene['scene_id'] if pre_scene else 'None'}")
        pre_data = np.zeros((1, TILE_SIZE, TILE_SIZE), np.uint16)
        has_pre = 0
    else:
        logger.debug(f"FID {fid}: PRE data OK - shape: {pre_data.shape}")
        has_pre = 1
    if post_data is None:
        logger.warning(f"FID {fid}: POST data es None - escena seleccionada: {post_scene['scene_id'] if post_scene else 'None'}")
        post_data = np.zeros((1, TILE_SIZE, TILE_SIZE), np.uint16)
        has_post = 0
    else:
        logger.debug(f"FID {fid}: POST data OK - shape: {post_data.shape}")
        has_post = 1
    if damage_label is None or (isinstance(damage_label, float) and math.isnan(damage_label)):
        damage_label = -1
    tile_path = output_dir / f"{fid}.npz"
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


def process_gpkg(gpkg_path, training_root, output_root, split):
    logger.info(f"Procesando {gpkg_path}")
    # Extraer nombre único de área/ciudad del archivo GPKG
    city_area = gpkg_path.stem.replace('_Training', '').replace('_training', '').replace('_Testing', '').replace('_testing', '').lower()
    output_dir = output_root / split / city_area
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = load_gpkg_with_fiona(gpkg_path)
    logger.info(f"Edificios en GPKG: {len(gdf)}")
    # Print columns for debugging Afghanistan areas
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
        fid_col = find_fid_column(gdf)  # Intentar de nuevo después de fiona
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
    logger.info(f"Fecha de activación: {activation_date}")
    # Load temporal interval from collection.json
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
    
    # Seleccionar las mejores escenas globales para mostrar fechas de referencia
    pre_scene_global, post_scene_global = select_best_scenes(scenes, activation_date)
    if pre_scene_global:
        logger.info(f"Fecha de escena PRE de referencia: {pre_scene_global['date']}")
    else:
        logger.info("No se encontró escena PRE de referencia")
    
    if post_scene_global:
        logger.info(f"Fecha de escena POST de referencia: {post_scene_global['date']}")
    else:
        logger.info("No se encontró escena POST de referencia")
    
    meta_data = []
    damage_col = 'damaged' if 'damaged' in gdf.columns else None
    logger.info(f"Columna de daño: {damage_col}")
    
    # No es necesario seleccionar las escenas aquí, se hará por edificio
    # pre_scene, post_scene = select_best_scenes(scenes, activation_date)
    
    crs_wkt = gdf.crs.to_wkt()
    tasks = []
    for idx, row in gdf.iterrows():
        fid = row[fid_col]
        damage = row[damage_col] if damage_col else -1
        
        # Pasamos todas las escenas y la fecha de activación a cada worker
        task = (row.geometry.wkb, (scenes, activation_date), crs_wkt, output_dir, fid, damage)
        logger.debug(f"Task structure: {task}")
        tasks.append(task)

    logger.debug(f"Tasks before executor.map: {tasks}")  # Nueva línea para registrar tasks

    # Configurar manejador de señales antes del pool
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    executor = None

    def signal_handler(signum, frame):
        if executor is not None:
            logger.info("\nCancelando procesos...")
            executor.shutdown(wait=False, cancel_futures=True)
        # Restaurar manejadores originales
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        sys.exit(1)

    try:
        # Instalar manejador de señales
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        with ProcessPoolExecutor(max_workers=12) as executor:
            try:
                for meta in tqdm(executor.map(process_tile_worker, tasks, chunksize=128), 
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
        # Restaurar manejadores originales
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)

    logger.info(f"Tiles generados para {gpkg_path}: {len(meta_data)}")
    return meta_data

def find_fid_column_with_fiona(gpkg_path, gdf):
    """Busca la columna 'id' o 'fid' usando Fiona como respaldo."""
    with fiona.open(gpkg_path) as src:
        schema_columns = src.schema['properties'].keys()
        logger.debug(f"Columnas en esquema Fiona: {list(schema_columns)}")
        if 'id' in schema_columns and 'id' not in gdf.columns:
            gdf['id'] = gdf.index
            logger.info("La columna 'id' detectada por Fiona se ha asignado al índice como 'id'.")
        elif 'fid' in schema_columns and 'fid' not in gdf.columns:
            gdf['id'] = gdf.index  # Usar 'id' en lugar de 'fid'
            logger.info("La columna 'fid' detectada por Fiona se ha asignado al índice como 'id'.")
    return gdf

def main():
    global P
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limitar número de GPKGs a procesar')
    parser.add_argument('--loglevel', type=str, default='INFO', help='Nivel de logging (DEBUG, INFO, WARNING)')
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    output_root = Path("preprocessed")
    if output_root.exists() and output_root.is_dir():
        shutil.rmtree(output_root)
    data_root = Path("data/international-charter-eo4ai-etq-challenge")
    training_annotations = data_root / "Training" / "Annotations"
    training_data = data_root / "Training"
    output_root.mkdir(exist_ok=True)
    # === CALCULAR PERCENTILES GLOBALES ===
    logger.info("Calculando percentiles globales por banda...")
    P = sample_patches_for_percentiles(training_data, sample_size=PERCENTILE_SAMPLE_SIZE)
    logger.info(f"Percentiles globales: {P}")
    gpkg_files = []
    for root, _, files in os.walk(training_annotations):
        for file in files:
            if file.lower().endswith(('training.gpkg', 'testing.gpkg')):
                gpkg_files.append(Path(root) / file)
    logger.info(f"Encontrados {len(gpkg_files)} archivos GPKG para procesar")
    for gpkg in gpkg_files:
        logger.info(f"  - {gpkg.name}")
    if args.limit:
        gpkg_files = gpkg_files[:args.limit]

    gpkg_files=gpkg_files[1:3]

    all_meta = []
    for gpkg_path in gpkg_files:
        fname = gpkg_path.name.lower()
        if fname.endswith("training.gpkg"):
            split = 'full'
        elif fname.endswith("testing.gpkg"):
            split = 'test'
        else:
            split = 'unknown'
        meta = process_gpkg(gpkg_path, training_data, output_root, split)
        all_meta.extend(meta)
    meta_df = pd.DataFrame(
        all_meta,
        columns=[
            "event",
            "gpkg",
            "fid",
            "tile_path",
            "damaged",
            "has_pre",
            "has_post",
            "split",
            "shift_x",
            "shift_y",
        ],
    )
    meta_df.to_csv(output_root / "meta.csv", index=False)
    logger.info(f"Procesamiento completo! Guardados {len(all_meta)} tiles")

if __name__ == "__main__":
    main()