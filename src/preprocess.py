import os
import re
import json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from pathlib import Path
from shapely.geometry import shape, box, mapping
from shapely import wkb
from datetime import datetime, timezone
from skimage.transform import resize
from tqdm import tqdm
import pandas as pd
import warnings
import shutil
from pyproj import Transformer, CRS
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import shapely
import sys
import logging
import argparse
from collections import Counter
import cv2

# Configuración global
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
ACTIVATION_DATES = {
    "Call_919_Turkey": datetime(2023, 2, 6, tzinfo=timezone.utc),
    "Call_1095_Myanmar": datetime(2025, 3, 23, tzinfo=timezone.utc),
    "Call_1075_China": datetime(2025, 1, 8, tzinfo=timezone.utc),
    "Call_920_Syria": datetime(2023, 2, 9, tzinfo=timezone.utc),
    "Call_965_Morocco": datetime(2023, 9, 10, tzinfo=timezone.utc),
    "Call_970_Afghanistan": datetime(2023, 10, 9, tzinfo=timezone.utc)
}

TILE_SIZE = 128
BUFFER_METERS = 10  # Buffer consistente en metros

# === GLOBAL PERCENTILES ===
PERCENTILE_SAMPLE_SIZE = 1000
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98
P = None  # Se calculará al inicio

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("preprocess")

def find_fid_column(gdf):
    fid_cols = [col for col in gdf.columns if col.lower().startswith('fid')]
    return fid_cols[0] if fid_cols else None

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
                if not (interval_start <= scene_date <= interval_end):
                    logger.warning(f"Escena {item_json.parent.name} ({scene_date}) fuera del intervalo temporal [{interval_start}, {interval_end}]")
                    continue
            # Read GSD
            gsd = read_gsd(item_json)
            if gsd:
                logger.debug(f"GSD para escena {item_json.parent.name}: {gsd} m/px")
            scenes.append({
                'scene_id': item_json.parent.name,
                'date': scene_date,
                'geometry': scene_geom,
                'path': item_json.parent,
                'bands': bands,
                'gsd': gsd
            })
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

def process_scene(scene, bbox, target_crs):
    global P
    if scene is None or not scene["bands"]:
        return None
    gsd = scene.get('gsd', TARGET_GSD)
    if gsd is None or gsd <= 0 or not np.isfinite(gsd):
        #logger.warning(f"GSD inválido ({gsd}) para escena {scene['scene_id']}, usando valor por defecto {TARGET_GSD}")
        gsd = TARGET_GSD
    scale = gsd / TARGET_GSD
    MAX_SCALE = 10.0
    if not np.isfinite(scale) or scale <= 0:
        #logger.warning(f"Factor de escala inválido ({scale}) para escena {scene['scene_id']}, usando 1.0")
        scale = 1.0
    elif scale > MAX_SCALE:
        scale = MAX_SCALE
    elif scale < 1/MAX_SCALE:
        scale = 1/MAX_SCALE
    band_arrays = []
    band_shapes = []
    band_names = list(scene["bands"].keys())
    # Primera pasada: leer todas las bandas y determinar el shape común
    for band_name, band_path in scene["bands"].items():
        try:
            with rasterio.open(band_path) as src:
                if src.crs != target_crs:
                    tf = Transformer.from_crs(target_crs, src.crs, always_xy=True).transform
                    minx, miny = tf(bbox[0], bbox[1])
                    maxx, maxy = tf(bbox[2], bbox[3])
                    bbox_t = (minx, miny, maxx, maxy)
                else:
                    bbox_t = bbox
                win = from_bounds(*bbox_t, transform=src.transform)
                if win.width == 0 or win.height == 0:
                    continue
                data = src.read(1, window=win, boundless=True, fill_value=0)
                if data.size == 0:
                    continue
                if not np.isclose(scale, 1.0):
                    try:
                        new_size = (int(data.shape[0] * scale), int(data.shape[1] * scale))
                        if all(s > 0 for s in new_size):
                            data = resize(
                                data,
                                new_size,
                                order=3,
                                mode="edge",
                                preserve_range=True,
                                anti_aliasing=True,
                            )
                        else:
                            #logger.warning(f"Tamaño de rescalado inválido {new_size} para banda {band_name}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error al rescalar banda {band_name}: {e}")
                        continue
                band_arrays.append(data)
                band_shapes.append(data.shape)
        except Exception as e:
            logger.warning(f"Error procesando banda {band_name}: {e}")
            continue
    if not band_arrays:
        return None
    shape_counts = Counter(band_shapes)
    target_shape = shape_counts.most_common(1)[0][0]
    normalized_arrays = []
    for idx, (data, shape) in enumerate(zip(band_arrays, band_shapes)):
        if shape != target_shape:
            try:
                data = resize(
                        data,
                        target_shape,
                        order=3,
                        mode="edge",
                        preserve_range=True,
                        anti_aliasing=True,
                )
            except Exception as e:
                logger.warning(f"Error normalizando shape de banda {idx}: {e}")
                continue
        # Redimensionar al tamaño final del tile
        try:
            data = resize(
                data,
                (TILE_SIZE, TILE_SIZE),
                order=3,
                mode="edge",
                preserve_range=True,
                anti_aliasing=True,
            )
            normalized_arrays.append(data)
        except Exception as e:
            logger.warning(f"Error en resize final de banda {idx}: {e}")
            continue
    if not normalized_arrays:
        return None
    # === NORMALIZACIÓN POR PERCENTILES ===
    stack = np.stack(normalized_arrays, axis=-1).astype(np.float32)
    for i, band_name in enumerate(band_names):
        if P and band_name in P:
            p2, p98 = P[band_name]
            stack[..., i] = np.clip(stack[..., i], p2, p98)
            stack[..., i] = (stack[..., i] - p2) / max(p98 - p2, 1e-8)
        else:
            mn = stack[..., i].min()
            mx = stack[..., i].max()
            stack[..., i] = (stack[..., i] - mn) / max(mx - mn, 1e-8)

    # --- REORDENAR A RGB SI EXISTEN ---
    desired_order = ['red', 'green', 'blue']
    ordered_arrays = []
    for band in desired_order:
        if band in band_names:
            idx = band_names.index(band)
            ordered_arrays.append(stack[..., idx])
        else:
            # Si falta la banda, rellena con ceros
            ordered_arrays.append(np.zeros_like(stack[..., 0]))
    # Si hay menos de 3 bandas, rellena hasta 3
    while len(ordered_arrays) < 3:
        ordered_arrays.append(np.zeros_like(stack[..., 0]))
    stack_rgb = np.stack(ordered_arrays, axis=-1)

    # Si quieres mantener las demás bandas, puedes concatenarlas después de RGB
    extra_bands = [i for i, b in enumerate(band_names) if b not in desired_order]
    if extra_bands:
        stack_extra = stack[..., extra_bands]
        stack_final = np.concatenate([stack_rgb, stack_extra], axis=-1)
    else:
        stack_final = stack_rgb

    try:
        return (stack_final.transpose(2, 0, 1) * 65535).round().astype(np.uint16)
    except Exception as e:
        logger.warning(f"Error en normalización final: {e}")
        return None

def process_tile_worker(args):
    wkb_geom, scenes, crs_wkt, output_dir, fid, damage = args
    geometry = wkb.loads(wkb_geom)
    crs = CRS.from_wkt(crs_wkt)
    return process_tile(geometry, scenes, crs, output_dir, fid, damage)

def process_tile(building, scenes, crs, output_dir, fid, damage_label):
    pre_scene, post_scene = scenes
    # Buffer consistente
    buffered_geom = buffer_geom(building, crs)
    bbox = buffered_geom.bounds
    pre_data  = process_scene(pre_scene,  bbox, crs) if pre_scene  else None
    post_data = process_scene(post_scene, bbox, crs) if post_scene else None
    if pre_data is None:
        pre_data  = np.zeros((1, TILE_SIZE, TILE_SIZE), np.uint16)
        has_pre   = 0
    else:
        has_pre   = 1
    if post_data is None:
        post_data = np.zeros((1, TILE_SIZE, TILE_SIZE), np.uint16)
        has_post  = 0
    else:
        has_post  = 1
    if damage_label is None or (isinstance(damage_label, float) and math.isnan(damage_label)):
        damage_label = -1
    tile_path = output_dir / f"{fid}.npz"
    np.savez_compressed(tile_path, pre=pre_data, post=post_data)
    return dict(fid=fid,
                tile_path=str(tile_path),
                damaged=int(damage_label),
                has_pre=has_pre,
                has_post=has_post)

def process_gpkg(gpkg_path, training_root, output_root, split):
    logger.info(f"Procesando {gpkg_path}")
    # Extraer nombre único de área/ciudad del archivo GPKG
    city_area = gpkg_path.stem.replace('_Training', '').replace('_training', '').replace('_Testing', '').replace('_testing', '').lower()
    output_dir = output_root / split / city_area
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(gpkg_path)
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
        logger.warning(f"No se encontró columna fid en {gpkg_path}, usando índice como fid.")
        gdf['fid'] = gdf.index
        fid_col = 'fid'
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
    meta_data = []
    damage_col = 'damaged' if 'damaged' in gdf.columns else None
    logger.info(f"Columna de daño: {damage_col}")
    pre_scene, post_scene = select_best_scenes(scenes, activation_date)
    crs_wkt = gdf.crs.to_wkt()
    tasks = []
    for idx, row in gdf.iterrows():
        fid = row[fid_col]
        damage = row[damage_col] if damage_col else -1
        if pre_scene is None and post_scene is None:
            logger.debug(f"FID {fid}: pre_scene y post_scene no encontrada")
            continue
        tasks.append((row.geometry.wkb, (pre_scene, post_scene), crs_wkt, output_dir, fid, damage))

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

    #gpkg_files=gpkg_files[:2]

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
    meta_df = pd.DataFrame(all_meta, columns=["event","gpkg","fid",
                                              "tile_path","damaged",
                                              "has_pre","has_post",
                                              "split"])
    meta_df.to_csv(output_root / "meta.csv", index=False)
    logger.info(f"Procesamiento completo! Guardados {len(all_meta)} tiles")

if __name__ == "__main__":
    main()