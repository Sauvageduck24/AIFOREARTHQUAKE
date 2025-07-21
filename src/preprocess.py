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

def find_scenes_for_aoi(aoi_geom, training_root):
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
            scenes.append({
                'scene_id': item_json.parent.name,
                'date': scene_date,
                'geometry': scene_geom,
                'path': item_json.parent,
                'bands': bands
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

def process_scene(scene, bbox, target_crs):
    if scene is None or not scene["bands"]:
        return None

    # Elegir siempre la MS-red si existe, si no la primera
    key0 = "red" if "red" in scene["bands"] else next(iter(scene["bands"]))
    with rasterio.open(scene["bands"][key0]) as src0:
        if src0.crs != target_crs:
            tf = Transformer.from_crs(target_crs, src0.crs, always_xy=True).transform
            minx, miny = tf(bbox[0], bbox[1])
            maxx, maxy = tf(bbox[2], bbox[3])
            bbox_t = (minx, miny, maxx, maxy)
        else:
            bbox_t = bbox
        win0 = from_bounds(*bbox_t, transform=src0.transform)
        if win0.width == 0 or win0.height == 0:
            return None        # el edificio no cae dentro de la escena

    band_arrays = []
    for band_name, band_path in scene["bands"].items():
        with rasterio.open(band_path) as src:
            data = src.read(1, window=win0, boundless=True, fill_value=0)
            if data.size == 0:
                continue
            band_arrays.append(
                resize(data, (TILE_SIZE, TILE_SIZE), order=1, mode="reflect",
                       preserve_range=True)
            )

    if not band_arrays:
        return None

    stack = np.stack(band_arrays, axis=-1).astype(np.float32)
    # Normalización vectorizada
    mn = stack.min(axis=(0, 1))
    mx = stack.max(axis=(0, 1))
    rng = np.maximum(mx - mn, 1e-8)
    norm = (stack - mn) / rng
    return (norm.transpose(2, 0, 1) * 65535).round().astype(np.uint16)

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
    if not activation_date:
        logger.warning(f"Fecha de activación desconocida para {event_name}")
        return []
    scenes = find_scenes_for_aoi(aoi_geom_wgs, call_dir)
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
    try:
        with ProcessPoolExecutor(max_workers=12) as executor:
            for meta in tqdm(executor.map(process_tile_worker, tasks, chunksize=128), total=len(tasks), desc=f"Procesando {gpkg_path.name}"):
                if meta:
                    meta['gpkg'] = gpkg_path.name
                    meta['event'] = event_name
                    meta['split'] = split
                    meta_data.append(meta)
    except KeyboardInterrupt:
        logger.info("Cancelado por el usuario. Terminando workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    logger.info(f"Tiles generados para {gpkg_path}: {len(meta_data)}")
    return meta_data

def main():
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

    #gpkg_files=gpkg_files[12:]
    
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