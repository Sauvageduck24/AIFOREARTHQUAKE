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

# Mapeo de eventos a países - actualizado
EVENT_TO_COUNTRY = {
    "Call_919_Turkey": "Turkey", 
    "Call_1095_Myanmar": "Myanmar", 
    "Call_1075_China": "China",
    "Call_920_Syria": "Syria", 
    "Call_965_Morocco": "Morocco", 
    "Call_970_Afghanistan": "Afghanistan"
}

# Mapeo de países a CRS - añadido desde visualizar_escenas.py
COUNTRY_TO_CRS = {
    "Turkey": "EPSG:32637", 
    "Myanmar": "EPSG:32646", 
    "China": "EPSG:32645",
    "Syria": "EPSG:32637", 
    "Morocco": "EPSG:32629", 
    "Afghanistan": "EPSG:32642"
}

# Fechas de activación por defecto (fallback)
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

def _read_rgb_bands_from_separate_files(scene_path, scene_id, out_shape):
    """Lee las bandas RGB desde archivos TIFF separados usando el JSON del satélite."""
    try:
        # Buscar el archivo JSON en diferentes ubicaciones posibles
        json_path = scene_path / f"{scene_id}.json"
        if not json_path.exists():
            # Intentar buscar en el directorio padre si no se encuentra
            json_path = scene_path.parent / f"{scene_id}.json"
            if not json_path.exists():
                return None, None
        
        logger.debug(f"Leyendo JSON desde: {json_path}")
        with open(json_path, 'r') as f:
            item_data = json.load(f)
        
        assets = item_data.get('assets', {})
        logger.debug(f"Assets encontrados en JSON: {list(assets.keys())}")
        rgb_assets = {}
        
        for key, asset in assets.items():
            if key in ['red', 'green', 'blue']:
                rgb_assets[key] = asset
            elif 'eo:bands' in asset:
                for band_info in asset['eo:bands']:
                    common_name = band_info.get('common_name', '').lower()
                    if common_name in ['red', 'green', 'blue']:
                        rgb_assets[common_name] = asset
                        break
        
        # Verificar qué bandas están disponibles
        available_bands = {}
        for color in ['red', 'green', 'blue']:
            if color in rgb_assets:
                asset = rgb_assets[color]
                # Construir la ruta relativa desde la carpeta de la escena
                asset_href = asset['href'].lstrip('./')
                base_asset_path = scene_path / asset_href
                
                # Si la ruta no existe, intentar buscar en diferentes ubicaciones
                if not base_asset_path.exists():
                    possible_paths = [
                        scene_path / "Optical_Calibration" / Path(asset_href).name,
                        scene_path.parent / "Optical_Calibration" / Path(asset_href).name,
                        scene_path / Path(asset_href).name,
                    ]
                    
                    for possible_path in possible_paths:
                        if possible_path.exists():
                            base_asset_path = possible_path
                            logger.debug(f"Archivo encontrado en ubicación alternativa: {possible_path}")
                            break
                
                # Buscar archivo _aligned.tif primero
                aligned_asset_path = base_asset_path.parent / f"{base_asset_path.stem}_aligned.tif"
                
                if aligned_asset_path.exists():
                    available_bands[color] = aligned_asset_path
                    logger.debug(f"Archivo alineado disponible para banda {color}: {aligned_asset_path.name}")
                elif base_asset_path.exists():
                    available_bands[color] = base_asset_path
                    logger.debug(f"Archivo original disponible para banda {color}: {base_asset_path.name}")
                else:
                    logger.warning(f"No se encontró archivo para banda {color}: {base_asset_path}")
            else:
                logger.warning(f"No se encontró asset para banda {color}")
        
        # Si no tenemos al menos una banda, fallar
        if not available_bands:
            logger.error("No se encontraron archivos para ninguna banda RGB")
            return None, None
        
        bands = []
        transform = None
        
        # Intentar leer las bandas disponibles
        for color in ['red', 'green', 'blue']:
            if color in available_bands:
                asset_path = available_bands[color]
                logger.debug(f"Leyendo banda {color} desde: {asset_path}")
                with rasterio.open(asset_path) as src:
                    if transform is None:
                        transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
                    band_data = src.read(1, out_shape=out_shape)
                    logger.debug(f"Banda {color} leída exitosamente, forma: {band_data.shape}, rango: [{band_data.min():.2f}, {band_data.max():.2f}]")
                    bands.append(band_data)
            else:
                # Si falta una banda, usar la primera disponible como sustituto
                first_available = list(available_bands.keys())[0]
                asset_path = available_bands[first_available]
                logger.debug(f"Usando banda {first_available} como sustituto para {color} desde: {asset_path}")
                with rasterio.open(asset_path) as src:
                    if transform is None:
                        transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
                    band_data = src.read(1, out_shape=out_shape)
                    logger.debug(f"Banda sustituta {first_available} leída exitosamente, forma: {band_data.shape}")
                    bands.append(band_data)
        
        return np.stack(bands, axis=-1), transform
        
    except Exception as e:
        logger.error(f"Error leyendo bandas RGB: {e}")
        return None, None

def _read_rgb_bands_from_tiff(tiff_path, out_shape):
    """Lee las bandas RGB desde un archivo TIFF multibanda."""
    try:
        with rasterio.open(tiff_path) as src:
            if src.count == 1:
                band_data = src.read(1, out_shape=out_shape)
                img_array = np.stack([band_data, band_data, band_data], axis=-1)
                transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
                return img_array, transform
            
            band_names = [desc.lower() if desc else '' for desc in src.descriptions]
            rgb_indices = {'red': None, 'green': None, 'blue': None}
            
            for i, name in enumerate(band_names):
                if any(color in name for color in ['red', 'r', 'b4', 'b04']):
                    rgb_indices['red'] = i
                elif any(color in name for color in ['green', 'g', 'b3', 'b03']):
                    rgb_indices['green'] = i
                elif any(color in name for color in ['blue', 'b', 'b2', 'b02']):
                    rgb_indices['blue'] = i
            
            if all(idx is None for idx in rgb_indices.values()) and src.count >= 3:
                rgb_indices = {'red': 0, 'green': 1, 'blue': 2}
            
            bands = []
            for color in ['red', 'green', 'blue']:
                band_idx = rgb_indices[color]
                if band_idx is not None and band_idx < src.count:
                    bands.append(src.read(band_idx + 1, out_shape=out_shape))
                else:
                    logger.warning(f"No se pudo encontrar banda {color} en {tiff_path}")
                    return None, None
            
            img_array = np.stack(bands, axis=-1)
            transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
            return img_array, transform
            
    except Exception as e:
        logger.error(f"Error leyendo TIFF: {e}")
        return None, None

def _normalize_for_display(img_array):
    """Normaliza un array para visualización usando percentiles."""
    logger.debug(f"Normalizando imagen con forma: {img_array.shape}, tipo: {img_array.dtype}")
    logger.debug(f"Rango original: [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
        logger.debug(f"Expandida a 3D, nueva forma: {img_array.shape}")
    
    # Manejar imágenes con 2 canales (crear un tercer canal)
    if img_array.shape[2] == 2:
        # Crear un tercer canal como promedio de los dos existentes
        third_channel = np.mean(img_array, axis=2, keepdims=True)
        img_array = np.concatenate([img_array, third_channel], axis=2)
        logger.debug(f"Convertida de 2 a 3 canales, nueva forma: {img_array.shape}")
    elif img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
        logger.debug(f"Repetida a 3 canales, nueva forma: {img_array.shape}")
    elif img_array.shape[2] > 3:
        # Si hay más de 3 canales, tomar solo los primeros 3
        img_array = img_array[:, :, :3]
        logger.debug(f"Reducida a 3 canales, nueva forma: {img_array.shape}")

    img_norm = np.zeros_like(img_array, dtype=np.float32)
    for i in range(img_array.shape[2]):
        band = img_array[:, :, i]
        if np.any(band > 0):
            p2, p98 = np.percentile(band[band > 0], (2, 98))
            logger.debug(f"Banda {i}: percentiles 2-98: [{p2:.2f}, {p98:.2f}]")
            if p98 > p2:
                img_norm[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                img_norm[:, :, i] = band.astype(np.float32) / (np.max(band) if np.max(band)>0 else 1)
        else:
            img_norm[:, :, i] = 0
            logger.debug(f"Banda {i}: todos los valores son 0")
    
    logger.debug(f"Imagen normalizada, rango final: [{img_norm.min():.2f}, {img_norm.max():.2f}]")
    return img_norm

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

def extract_earthquake_dates_from_collection(collection_path):
    """Extrae las fechas de terremotos desde el collection.json y carpetas de archivos."""
    earthquake_dates = {}
    
    try:
        with open(collection_path, 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
        
        # Buscar fechas en los nombres de carpetas y archivos
        training_root = collection_path.parent
        
        # Si estamos en una carpeta de evento específica, buscar en esa carpeta
        if training_root.name.startswith('Call_'):
            folders_to_check = [training_root]
        else:
            # Si estamos en el directorio raíz, buscar en todas las carpetas de eventos
            folders_to_check = [folder for folder in training_root.iterdir() 
                              if folder.is_dir() and folder.name.startswith('Call_')]
        
        for folder in folders_to_check:
            if folder.is_dir() and folder.name.startswith('Call_'):
                # Extraer país del nombre de la carpeta
                country = EVENT_TO_COUNTRY.get(folder.name, folder.name)
                
                # Buscar fechas en el nombre de la carpeta (formato: 20230626)
                date_match = re.search(r'(\d{8})', folder.name)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        earthquake_date = datetime.strptime(date_str, '%Y%m%d')
                        earthquake_dates[country] = earthquake_date
                        logger.info(f"Fecha de terremoto detectada para {country}: {earthquake_date.strftime('%Y-%m-%d')}")
                        continue
                    except ValueError:
                        pass
                
                # Si no se encuentra en el nombre de la carpeta, buscar en los archivos JSON
                json_files = list(folder.glob("*.json"))
                for json_file in json_files:
                    if json_file.name != "collection.json":
                        try:
                            with open(json_file, 'r') as f:
                                item_data = json.load(f)
                            
                            # Buscar fecha en el ID del archivo (formato: 20230626)
                            scene_id = item_data.get('id', '')
                            date_match = re.search(r'(\d{8})', scene_id)
                            if date_match:
                                date_str = date_match.group(1)
                                try:
                                    earthquake_date = datetime.strptime(date_str, '%Y%m%d')
                                    earthquake_dates[country] = earthquake_date
                                    logger.info(f"Fecha de terremoto detectada para {country} desde {json_file.name}: {earthquake_date.strftime('%Y-%m-%d')}")
                                    break
                                except ValueError:
                                    pass
                            
                            # Buscar fecha en las propiedades datetime
                            properties = item_data.get('properties', {})
                            datetime_str = properties.get('datetime')
                            if datetime_str:
                                try:
                                    earthquake_date = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                                    earthquake_dates[country] = earthquake_date
                                    logger.info(f"Fecha de terremoto detectada para {country} desde propiedades: {earthquake_date.strftime('%Y-%m-%d')}")
                                    break
                                except ValueError:
                                    pass
                            
                            # Buscar fecha en propiedades adicionales
                            for date_key in ['start_datetime', 'end_datetime', 'created', 'updated']:
                                if date_key in properties:
                                    try:
                                        earthquake_date = datetime.fromisoformat(properties[date_key].replace('Z', '+00:00'))
                                        earthquake_dates[country] = earthquake_date
                                        logger.info(f"Fecha de terremoto detectada para {country} desde {date_key}: {earthquake_date.strftime('%Y-%m-%d')}")
                                        break
                                    except ValueError:
                                        pass
                        except Exception as e:
                            continue
                            
    except Exception as e:
        logger.error(f"Error extrayendo fechas de terremotos: {e}")
    
    return earthquake_dates

def get_activation_date(call_dir: Path) -> datetime | None:
    # 1️⃣ Intenta leer collection.json con nuevo método
    coll_json = call_dir / "collection.json"
    if coll_json.exists():
        earthquake_dates = extract_earthquake_dates_from_collection(coll_json)
        country = EVENT_TO_COUNTRY.get(call_dir.name, call_dir.name)
        if country in earthquake_dates:
            return earthquake_dates[country].replace(tzinfo=timezone.utc)
        
        # Fallback al método original
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
            # Estimar desde el primer band como fallback
            calib_dir = item_json.parent / "Optical_Calibration"
            band_files = list(calib_dir.glob('r-*.tif'))
            if band_files:
                with rasterio.open(band_files[0]) as src:
                    gsd_from_raster = abs(src.transform.a)
                    if 0.1 <= gsd_from_raster <= 100:  # Si el GSD del raster es razonable
                        gsd = gsd_from_raster
                    else:
                        gsd = None  # Se manejará en process_scene
    
    return gsd

def find_scenes_for_aoi(aoi_geom, training_root, interval_start=None, interval_end=None, satellites=None):
    """Encuentra escenas usando collection.json y metadatos actualizados."""
    training_root = Path(training_root)
    
    # Procesar lista de satélites
    if satellites is None:
        satellites = ['wv', 'ds']
    elif isinstance(satellites, str):
        satellites = [s.strip().lower() for s in satellites.split(',')]
    
    logger.info(f"Filtrando escenas por satélites: {satellites}")
    
    # Buscar collection.json en diferentes ubicaciones
    collection_path = None
    possible_collection_paths = [
        training_root / "collection.json",
        training_root / "Training" / "collection.json",
    ]
    
    # También buscar en cada carpeta de evento
    for event_dir in training_root.glob("Call_*"):
        if event_dir.is_dir():
            possible_collection_paths.append(event_dir / "collection.json")
    
    for path in possible_collection_paths:
        if path.exists():
            collection_path = path
            logger.info(f"Encontrado collection.json en: {collection_path}")
            break
    
    if not collection_path:
        logger.warning(f"No se encontró collection.json en ninguna ubicación esperada")
        return []

    # Extraer fechas de terremotos
    earthquake_dates = extract_earthquake_dates_from_collection(collection_path)

    with open(collection_path, 'r', encoding='utf-8') as f:
        collection_data = json.load(f)

    all_item_data = []
    item_links = [link['href'] for link in collection_data.get('links', []) if link.get('rel') == 'item']
    logger.info(f"Encontrados {len(item_links)} items en collection.json")
    
    # Si no hay items en collection.json, buscar archivos JSON directamente
    if not item_links:
        logger.info("No se encontraron items en collection.json, buscando archivos JSON directamente...")
        for event_dir in training_root.glob("Call_*"):
            if event_dir.is_dir():
                for json_file in event_dir.glob("*-calibrated/*.json"):
                    if json_file.name != "collection.json":
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                item_data = json.load(f)
                                item_data['__item_path'] = json_file
                                all_item_data.append(item_data)
                        except Exception as e:
                            logger.warning(f"Error leyendo {json_file}: {e}")
        logger.info(f"Encontrados {len(all_item_data)} archivos JSON directamente")
    
    for item_href in item_links:
        item_path = (collection_path.parent / item_href).resolve()
        if item_path.exists():
            try:
                with open(item_path, 'r', encoding='utf-8') as f:
                    item_data = json.load(f)
                    item_data['__item_path'] = item_path
                    all_item_data.append(item_data)
            except Exception as e:
                logger.error(f"Error leyendo item JSON {item_path}: {e}")
        else:
            logger.warning(f"Item no encontrado: {item_path}")

    # Agrupar escenas MS y PN del mismo satélite
    grouped_scenes = {}
    logger.info(f"Agrupando {len(all_item_data)} items...")
    
    for item in all_item_data:
        scene_id = item.get('id', '')
        base_name = scene_id
        
        # Mejor lógica de agrupamiento
        if 'WV03N28_549166E087' in scene_id:
            base_name = 'WV03N28_549166E087'
        else:
            if re.search(r"MS\d{2}", scene_id):
                base_name = scene_id.split('_MS')[0]
            elif re.search(r"PN\d{2}", scene_id):
                base_name = scene_id.split('_PN')[0]
            if base_name.endswith('-calibrated'):
                base_name = base_name[:-11]
        
        if base_name not in grouped_scenes:
            grouped_scenes[base_name] = {}
        
        if re.search(r"MS\d{2}", scene_id):
            grouped_scenes[base_name]['MS'] = item
        elif re.search(r"PN\d{2}", scene_id):
            grouped_scenes[base_name]['PN'] = item
        else:
            grouped_scenes[base_name]['OTRO'] = item

    # Procesar escenas agrupadas
    final_items_to_process = []
    for base_name, items in grouped_scenes.items():
        is_worldview = 'wv' in base_name.lower() or 'worldview' in base_name.lower()
        if is_worldview:
            if 'MS' in items:
                final_items_to_process.append(items['MS'])
            else:
                logger.warning(f"No se encontró escena MS para WorldView {base_name}")
        else:
            final_items_to_process.extend(items.values())

    scenes = []
    for item_data in final_items_to_process:
        try:
            scene_id = item_data.get('id')
            properties = item_data.get('properties', {})
            assets = item_data.get('assets', {})
            item_path = item_data['__item_path']

            scene_date_str = properties.get('datetime')
            scene_date = datetime.fromisoformat(scene_date_str.replace('Z', '+00:00')) if scene_date_str else None

            # Filtrar por satélite
            scene_id_lower = scene_id.lower()
            platform = properties.get('platform', '').lower()
            
            # Mapeo de abreviaciones de satélites a patrones de búsqueda
            satellite_patterns = {
                'wv': ['wv', 'worldview', 'world view'],
                'ds': ['ds_phr', 'ds-phr', 'pleiades', 'pleiade'],
                'pleiades': ['pleiades', 'pleiade', 'ds_phr', 'ds-phr'],
                'geoeye': ['geoeye', 'geo eye', 'ge'],
                'ge': ['geoeye', 'geo eye', 'ge']
            }
            
            # Verificar si la escena corresponde a alguno de los satélites solicitados
            satellite_match = False
            for satellite in satellites:
                patterns = satellite_patterns.get(satellite, [satellite])
                for pattern in patterns:
                    if pattern in scene_id_lower or pattern in platform:
                        satellite_match = True
                        logger.debug(f"Escena {scene_id} coincide con satélite '{satellite}' (patrón: {pattern})")
                        break
                if satellite_match:
                    break
            
            if not satellite_match:
                logger.debug(f"Escena {scene_id} descartada - no coincide con satélites solicitados: {satellites}")
                continue

            # Buscar overview asset con mejor lógica
            overview_asset = None
            overview_path = None
            
            # Priorizar assets específicos
            for key, asset in assets.items():
                if any(name in key.lower() for name in ['overview', 'visual', 'rgb', 'natural']) and 'href' in asset:
                    overview_asset = asset
                    break
            
            # Si no se encuentra, buscar cualquier asset con roles visual
            if not overview_asset:
                for key, asset in assets.items():
                    if 'visual' in asset.get('roles', []) and 'href' in asset:
                        overview_asset = asset
                        break
            
            # Si aún no se encuentra, buscar cualquier asset con href
            if not overview_asset:
                for key, asset in assets.items():
                    if 'href' in asset and not key.startswith('_'):
                        overview_asset = asset
                        break
            
            if not overview_asset:
                logger.warning(f"No se encontró asset de overview para {scene_id}")
                continue

            # Construir la ruta de la escena
            scene_folder = item_path.parent
            asset_href = overview_asset['href']
            asset_filename = Path(asset_href).name
            asset_stem = Path(asset_href).stem
            asset_suffix = Path(asset_href).suffix
            
            # Crear nombres de archivo con y sin "_aligned"
            aligned_filename = f"{asset_stem}_aligned{asset_suffix}"
            
            # Buscar archivos priorizando _aligned
            possible_base_paths = [
                scene_folder / asset_href,
                scene_folder / asset_filename,
                scene_folder / "Optical_Calibration" / asset_filename,
            ]
            
            overview_path = None
            
            # Primero buscar versiones _aligned
            for base_path in possible_base_paths:
                aligned_path = base_path.parent / aligned_filename
                if aligned_path.exists():
                    overview_path = aligned_path.resolve()
                    logger.debug(f"Archivo alineado encontrado: {overview_path}")
                    break
            
            # Si no se encuentra _aligned, buscar archivos originales
            if overview_path is None:
                for possible_path in possible_base_paths:
                    if possible_path.exists():
                        overview_path = possible_path.resolve()
                        break
            
            if overview_path is None:
                logger.warning(f"No se encontró archivo para {scene_id}")
                continue

            # Determinar el path de la escena
            if "Optical_Calibration" in str(overview_path):
                scene_path = overview_path.parent.parent
            else:
                scene_path = overview_path.parent

            # Extraer información geoespacial
            scene_geom = None
            scene_crs = None
            scene_gsd = None
            
            try:
                with rasterio.open(overview_path) as src:
                    scene_crs = str(src.crs) if src.crs else None
                    if src.transform:
                        scene_gsd = abs(src.transform[0])
                    
                    bounds = src.bounds
                    scene_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                    
                    # Reprojectar a WGS84 si es necesario
                    if src.crs and src.crs != 'EPSG:4326':
                        try:
                            transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                            scene_geom = shapely.ops.transform(transformer.transform, scene_geom)
                        except Exception as e:
                            logger.warning(f"Error reproyectando geometría: {e}")
                    
            except Exception as e:
                logger.warning(f"Error extrayendo información del TIFF {overview_path}: {e}")
                scene_geom = shape(item_data['geometry']) if 'geometry' in item_data else None
                scene_crs = f"EPSG:{properties['proj:epsg']}" if properties.get('proj:epsg') is not None else None
                scene_gsd = properties.get('gsd')

            # Verificar intersección con AOI
            if not aoi_geom.intersects(scene_geom):
                continue

            # Leer bandas con soporte para archivos _aligned
            calib_dir = scene_path / "Optical_Calibration"
            if not calib_dir.exists():
                continue
                
            bands = {}
            for band_file in calib_dir.glob("r-*.tif"):
                band_name = band_file.stem.split('-')[-1]
                
                # Buscar versión _aligned primero
                aligned_band_file = band_file.parent / f"{band_file.stem}_aligned.tif"
                if aligned_band_file.exists():
                    bands[band_name] = aligned_band_file
                    logger.debug(f"Usando banda alineada: {aligned_band_file.name}")
                else:
                    bands[band_name] = band_file

            # Añadir pan si existe
            pan_file = calib_dir / "r-pan.tif"
            aligned_pan_file = calib_dir / "r-pan_aligned.tif"
            if aligned_pan_file.exists():
                bands['pan'] = aligned_pan_file
            elif pan_file.exists():
                bands['pan'] = pan_file

            # Filter by temporal interval
            if interval_start and interval_end and scene_date:
                if scene_date < interval_start:
                    continue
                
                if scene_date > interval_end + timedelta(hours=MAX_EXTENT_OVERLAP_HRS):
                    logger.warning(f"{scene_id} fuera de intervalo "
                                 f"({scene_date} > {interval_end})  — descartado")
                    continue
                elif scene_date > interval_end:
                    hours_over = (scene_date - interval_end).total_seconds() / 3600
                    logger.info(f"{scene_id} {scene_date} a {hours_over:.1f} h "
                               f"del límite; lo acepto por margen +{MAX_EXTENT_OVERLAP_HRS} h")

            scene_info = {
                'scene_id': scene_id,
                'date': scene_date,
                'geometry': scene_geom,
                'path': scene_path,
                'bands': bands,
                'gsd': scene_gsd,
                'cloud_cover': properties.get('eo:cloud_cover'),
                'sat:off_nadir': properties.get('sat:off_nadir'),
                'earthquake_dates': earthquake_dates
            }
            
            scenes.append(scene_info)
            logger.info(f"Escena añadida: {scene_id} con {len(bands)} bandas")
            
        except Exception as e:
            logger.warning(f"Error procesando escena {item_data.get('id', 'unknown')}: {e}")
            continue
    
    logger.info(f"Total de escenas procesadas: {len(scenes)}")
    return scenes

def select_best_scenes(scenes, activation_date):
    """Selecciona las mejores escenas pre y post-terremoto basándose en fecha, cobertura de nubes y ángulo off-nadir."""
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
    
    # Función de scoring que considera múltiples factores
    def score_scene(scene, ref_date):
        """Calcula un score para la escena basado en múltiples factores."""
        temporal_dist = abs(ref_date - scene['date']).total_seconds() / 3600  # horas
        cloud_cover = scene.get('cloud_cover')
        off_nadir = scene.get('sat:off_nadir')
        
        # Manejar valores None y normalizar factores (valores más bajos son mejores)
        temporal_score = temporal_dist / 24  # Normalizar por días
        cloud_score = (cloud_cover / 100) if cloud_cover is not None else 0.5  # 50% por defecto
        nadir_score = (off_nadir / 45) if off_nadir is not None else 0.67  # 30° por defecto normalizado
        
        # Peso combinado (temporal es más importante)
        total_score = (temporal_score * 0.6) + (cloud_score * 0.25) + (nadir_score * 0.15)
        return total_score
    
    # Ordenar por score (mejor score = menor valor)
    pre_scenes.sort(key=lambda x: score_scene(x, activation_date))
    post_scenes.sort(key=lambda x: score_scene(x, activation_date))
    
    pre_scene = pre_scenes[0] if pre_scenes else None
    post_scene = post_scenes[0] if post_scenes else None
    
    # Log de selección para debugging
    if pre_scene:
        logger.info(f"Mejor escena PRE seleccionada: {pre_scene['scene_id']} "
                   f"(fecha: {pre_scene['date']}, nubes: {pre_scene.get('cloud_cover', 'N/A')}%, "
                   f"off-nadir: {pre_scene.get('sat:off_nadir', 'N/A')}°)")
    if post_scene:
        logger.info(f"Mejor escena POST seleccionada: {post_scene['scene_id']} "
                   f"(fecha: {post_scene['date']}, nubes: {post_scene.get('cloud_cover', 'N/A')}%, "
                   f"off-nadir: {post_scene.get('sat:off_nadir', 'N/A')}°)")
    
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
        logger.warning(f"GSD demasiado pequeño: {gsd} para escena {scene['scene_id']}, usando TARGET_GSD = {TARGET_GSD}")
        gsd = TARGET_GSD
    elif gsd > 100:
        logger.warning(f"GSD demasiado grande: {gsd} para escena {scene['scene_id']}, usando TARGET_GSD = {TARGET_GSD}")
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
        logger.warning(f"Scale extremadamente pequeño detectado: {scale:.10f} para escena {scene['scene_id']}, forzando scale = 1.0")
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
                    logger.warning(f"Datos vacíos para banda {band_name} en escena {scene['scene_id']}")
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
                        logger.warning(f"Máscara vacía para banda {band_name} en escena {scene['scene_id']}")
                        continue

                    # Validar que data tenga dimensiones válidas antes de calcular target_h/target_w
                    if data.shape[0] == 0 or data.shape[1] == 0:
                        logger.warning(f"Datos con dimensiones inválidas {data.shape} para banda {band_name} en escena {scene['scene_id']}")
                        continue

                    # Re-muestrear la máscara con el MISMO pipeline que el RGB
                    target_h = int(data.shape[0] * scale)
                    target_w = int(data.shape[1] * scale)
                    
                    # Validar que las dimensiones objetivo sean válidas
                    if target_h <= 0 or target_w <= 0:
                        logger.warning(f"Dimensiones objetivo inválidas {target_h}x{target_w} para banda {band_name} en escena {scene['scene_id']}")
                        continue

                    if scale != 1.0:
                        # Validar dimensiones antes de resize
                        if target_h <= 0 or target_w <= 0:
                            logger.warning(f"Dimensiones inválidas para resize: {target_h}x{target_w}")
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
                        logger.warning(f"Máscara vacía después de resize para banda {band_name}")
                        continue

                    mask = cv2.resize(
                        mask_full.astype(np.uint8),
                        (TILE_SIZE, TILE_SIZE),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)

                # Validar que data no esté vacía
                if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
                    logger.warning(f"Datos vacíos para banda {band_name} antes del resize")
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

    # Índices de bandas que NO son red/green/blue/pan
    # Incluir pan en las bandas extra ya que tiene alta resolución
    extra_bands = ["pan", "nir", "swir1", "swir2", "coastal", "yellow", "rededge"]
    extra_idx = [i for i, b in enumerate(band_names) if b not in desired_order and b in extra_bands]

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

    # Sin alineación entre escenas pre/post
    shift_xy = (0.0, 0.0)

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

def process_gpkg(gpkg_path, training_root, output_root, split, max_workers=12, satellites=None):
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
    scenes = find_scenes_for_aoi(aoi_geom_wgs, call_dir, interval_start, interval_end, satellites)
    logger.info(f"Escenas encontradas: {len(scenes)}")
    if not scenes:
        logger.warning(f"No se encontraron escenas para {gpkg_path}")
        logger.info(f"Call directory: {call_dir}")
        logger.info(f"AOI geometry bounds: {aoi_geom_wgs.bounds}")
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

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    parser.add_argument('--data-root', type=str, default='data/international-charter-eo4ai-etq-challenge', 
                       help='Ruta al directorio raíz de datos')
    parser.add_argument('--output-dir', type=str, default='preprocessed', 
                       help='Directorio de salida para datos preprocesados')
    parser.add_argument('--tile-size', type=int, default=224, 
                       help='Tamaño de los tiles de salida')
    parser.add_argument('--buffer-meters', type=int, default=200, 
                       help='Buffer en metros alrededor de cada edificio')
    parser.add_argument('--max-workers', type=int, default=12, 
                       help='Número máximo de workers para procesamiento paralelo')
    parser.add_argument('--satellites', type=str, default='wv,ds', 
                       help='Satélites a usar (separados por coma). Opciones: wv,ds,pleiades,geoeye,ge. Por defecto: wv,ds')
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    
    # Mostrar información sobre los satélites seleccionados
    satellites_list = [s.strip().lower() for s in args.satellites.split(',')]
    logger.info(f"Satélites seleccionados: {satellites_list}")
    
    # Actualizar variables globales con argumentos
    global TILE_SIZE, BUFFER_METERS
    TILE_SIZE = args.tile_size
    BUFFER_METERS = args.buffer_meters
    
    output_root = Path(args.output_dir)
    if output_root.exists() and output_root.is_dir():
        shutil.rmtree(output_root)
    data_root = Path(args.data_root)
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
        meta = process_gpkg(gpkg_path, training_data, output_root, split, args.max_workers, args.satellites)
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