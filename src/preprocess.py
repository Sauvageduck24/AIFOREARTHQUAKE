# --- Configuración de threads para evitar conflictos en multiprocesamiento ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Configuración de GDAL para optimización de I/O ---
os.environ['GDAL_CACHEMAX'] = '1024'  # 1GB cache para GDAL

import argparse
import json
import logging
import math
import re
import shutil
import signal
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock

import cv2
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import shapely
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.windows import from_bounds
from scipy.ndimage import shift as ndi_shift
from shapely import wkb
from shapely.geometry import box, mapping, shape
from skimage.registration import phase_cross_correlation
from skimage.transform import resize
from tqdm import tqdm



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

NO_UPSAMPLE = False  # evita escalar hacia arriba
MAX_EXTENT_OVERLAP_HRS = 72000  # acepta escenas +720 h fuera del extent.temporal.end

# === GLOBAL PERCENTILES ===
PERCENTILE_SAMPLE_SIZE = 2000  # Más muestras para percentiles más precisos
PERCENTILE_LOW = 1.0   # Percentil bajo para normalización
PERCENTILE_HIGH = 99.0 # Percentil alto para normalización
P = None  # Se calculará al inicio
SCENES = None  # Se pasará a los workers

# Cache para datasets de rasterio en workers
RASTER_CACHE = {}

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("preprocess")

# Configurar logging para que no interfiera con tqdm
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Asegurar que el mensaje termine con nueva línea para separarlo de la barra
            if not msg.endswith('\n'):
                msg += '\n'
            tqdm.write(msg, end='')
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

# Aplicar el handler personalizado
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(TqdmLoggingHandler())
logger.setLevel(logging.INFO)



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
    logger.warning(f"Error buscando las fechas, se usan las fechas de activación de los terremotos por defecto: {ACTIVATION_DATES.get(call_dir.name)}")
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
    
    # Detectar sensor para usar GSD por defecto apropiado
    sensor = item.get('properties', {}).get('platform', 'unknown').lower()
    
    # GSDs por defecto basados en especificaciones técnicas de cada sensor
    sensor_default_gsd = {
        'worldview-2': 2.0,  # MS: ~2.0m
        'worldview-3': 1.24, # MS: ~1.24m
        'pleiades': 2.0,     # MS: ~2.0m
        'pleiades-1a': 2.0,
        'pleiades-1b': 2.0,
        'kompsat-3a': 2.2,   # MS: ~2.2m
        'gaofen-2': 4.0,     # MS: ~4.0m
        'geoeye-1': 2.0,     # MS: ~2.0m
        'unknown': 2.0       # Fallback conservador
    }
    
    gsd = None
    
    # PRIMERA OPCIÓN: Leer GSD directamente del TIFF (más confiable)
    calib_dir = item_json.parent / "Optical_Calibration"
    band_files = list(calib_dir.glob('r-*.tif'))
    if band_files:
        try:
            with rasterio.open(band_files[0]) as src:
                pixel_size = abs(src.transform.a)
                
                # Convertir unidades según el CRS
                if src.crs:
                    crs_obj = CRS.from_string(str(src.crs))
                    
                    # Si está en coordenadas geográficas (grados), convertir a metros
                    if crs_obj.is_geographic:
                        # Aproximación: 1 grado ≈ 111320 metros en el ecuador
                        # Más preciso sería usar la latitud, pero esta es una buena aproximación
                        gsd_from_tiff = pixel_size * 111320
                        logger.debug(f"CRS geográfico detectado, convirtiendo {pixel_size}° a {gsd_from_tiff:.2f}m")
                    else:
                        # Ya está en unidades proyectadas (probablemente metros)
                        gsd_from_tiff = pixel_size
                        logger.debug(f"CRS proyectado detectado, GSD: {gsd_from_tiff}m")
                else:
                    # Sin CRS, asumir que ya está en metros
                    gsd_from_tiff = pixel_size
                    logger.warning(f"Sin CRS detectado, asumiendo GSD en metros: {gsd_from_tiff}")
                
                if 0.1 <= gsd_from_tiff <= 100:  # Validar que sea razonable
                    gsd = gsd_from_tiff
                    logger.debug(f"GSD leído del TIFF: {gsd:.2f}m para sensor {sensor}")
                else:
                    logger.warning(f"GSD del TIFF inválido ({gsd_from_tiff:.6f}) para sensor {sensor}")
        except Exception as e:
            logger.warning(f"Error leyendo GSD del TIFF: {e}")
    
    # SEGUNDA OPCIÓN: Usar GSD del JSON solo si el TIFF no funcionó
    if gsd is None:
        gsd_json = item['properties'].get('gsd')
        if gsd_json is not None:
            if 0.1 <= gsd_json <= 100:  # Validar que sea razonable
                gsd = gsd_json
                logger.debug(f"GSD leído del JSON: {gsd}m para sensor {sensor}")
            else:
                logger.warning(f"GSD del JSON inválido ({gsd_json}) para sensor {sensor}")
    
    # TERCERA OPCIÓN: Usar GSD por defecto del sensor
    if gsd is None or not np.isfinite(gsd) or gsd <= 0:
        gsd = sensor_default_gsd.get(sensor, 2.0)
        logger.warning(f"Usando GSD por defecto para sensor {sensor}: {gsd}m")
    
    return gsd

def determine_optimal_target_gsd(scenes):
    """
    Determina automáticamente el TARGET_GSD óptimo basándose en las características de las escenas.
    
    Estrategia:
    1. Analizar el GSD de todas las escenas disponibles
    2. Usar el GSD mínimo para preservar máxima resolución
    3. Considerar el tipo de sensor y sus capacidades
    4. Balancear calidad vs rendimiento
    
    Args:
        scenes: Lista de escenas con información de GSD y sensor
    
    Returns:
        TARGET_GSD óptimo en metros
    """
    if not scenes:
        logger.warning("No hay escenas disponibles, usando TARGET_GSD por defecto: 0.5m")
        return 0.5
    
    # Extraer GSDs de todas las escenas
    gsds = []
    sensors = []
    
    for scene in scenes:
        if scene.get('gsd') and np.isfinite(scene['gsd']) and scene['gsd'] > 0:
            gsds.append(scene['gsd'])
            # Extraer información del sensor
            scene_id = scene.get('scene_id', '').lower()
            if any(sat in scene_id for sat in ['wv', 'worldview']):
                sensors.append('worldview')
            elif any(sat in scene_id for sat in ['pleiades', 'ds_phr']):
                sensors.append('pleiades')
            elif any(sat in scene_id for sat in ['kompsat', 'k3']):
                sensors.append('kompsat')
            elif any(sat in scene_id for sat in ['gaofen', 'gf']):
                sensors.append('gaofen')
            elif any(sat in scene_id for sat in ['geoeye', 'ge']):
                sensors.append('geoeye')
            else:
                sensors.append('unknown')
    
    if not gsds:
        logger.warning("No se pudieron extraer GSDs válidos, usando TARGET_GSD por defecto: 0.5m")
        return 0.5
    
    # Estadísticas de GSD
    min_gsd = min(gsds)
    max_gsd = max(gsds)
    median_gsd = np.median(gsds)
    mean_gsd = np.mean(gsds)
    
    logger.info(f"Análisis de GSD de {len(gsds)} escenas:")
    logger.info(f"  Mínimo: {min_gsd:.2f}m, Máximo: {max_gsd:.2f}m")
    logger.info(f"  Mediana: {median_gsd:.2f}m, Media: {mean_gsd:.2f}m")
    
    # === NUEVA ESTRATEGIA: TARGET_GSD CONSERVADOR PARA EVITAR UPSAMPLE EXCESIVO ===
    # Usar un TARGET_GSD conservador para evitar upsample excesivo
    if sensors:
        sensor_counts = Counter(sensors)
        most_common_sensor = sensor_counts.most_common(1)[0][0]
        logger.info(f"Sensor predominante: {most_common_sensor}")
        
        # TARGET_GSD conservador por sensor para evitar upsample excesivo
        if most_common_sensor == 'worldview':
            target_gsd = max(min_gsd, 1.0)  # Mínimo 1.0m para WorldView
            logger.info(f"WorldView detectado: TARGET_GSD conservador = {target_gsd:.2f}m")
        elif most_common_sensor == 'pleiades':
            target_gsd = max(min_gsd, 0.5)  # Mínimo 0.5m para Pleiades
            logger.info(f"Pleiades detectado: TARGET_GSD conservador = {target_gsd:.2f}m")
        elif most_common_sensor == 'kompsat':
            target_gsd = max(min_gsd, 1.0)  # Mínimo 1.0m para Kompsat
            logger.info(f"Kompsat detectado: TARGET_GSD conservador = {target_gsd:.2f}m")
        else:
            target_gsd = max(min_gsd, 1.0)  # Mínimo 1.0m para otros sensores
            logger.info(f"Otro sensor detectado: TARGET_GSD conservador = {target_gsd:.2f}m")
    else:
        target_gsd = max(min_gsd, 1.0)  # Mínimo 1.0m por defecto
        logger.info(f"Sin sensor predominante: TARGET_GSD conservador = {target_gsd:.2f}m")
    
    # Límite superior para evitar tiles demasiado grandes
    if target_gsd < 0.3:
        target_gsd = 0.3
        logger.info(f"Limitando TARGET_GSD a 0.3m para evitar tiles excesivamente grandes")
    elif target_gsd > 4.0:
        target_gsd = 4.0
        logger.info(f"Limitando TARGET_GSD a 4.0m para evitar tiles demasiado pequeños")
    
    logger.info(f"TARGET_GSD final determinado automáticamente: {target_gsd:.2f}m (estrategia conservadora)")
    return target_gsd

def find_scenes_for_aoi(aoi_geom, training_root, interval_start=None, interval_end=None, satellites=None):
    """Encuentra escenas usando collection.json y metadatos actualizados."""
    training_root = Path(training_root)
    
    # Procesar lista de satélites
    if satellites is None:
        satellites = ['all']
    elif isinstance(satellites, str):
        satellites = [s.strip().lower() for s in satellites.split(',')]
    
    # logger.info(f"Filtrando escenas por satélites: {satellites}")
    
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
            # logger.info(f"Encontrado collection.json en: {collection_path}")
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
    # logger.info(f"Encontrados {len(item_links)} items en collection.json")
    
    # Si no hay items en collection.json, buscar archivos JSON directamente
    if not item_links:
        # logger.info("No se encontraron items en collection.json, buscando archivos JSON directamente...")
        for event_dir in training_root.glob("Call_*"):
            if event_dir.is_dir():
                # Buscar archivos JSON directamente en las carpetas *-calibrated
                for json_file in event_dir.glob("*-calibrated/*.json"):
                    if json_file.name != "collection.json":
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                item_data = json.load(f)
                                item_data['__item_path'] = json_file
                                all_item_data.append(item_data)
                        except Exception as e:
                            logger.warning(f"Error leyendo {json_file}: {e}")
        # logger.info(f"Encontrados {len(all_item_data)} archivos JSON directamente")
    
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

    # Agrupar escenas MS y PN del mismo satélite por fecha y coordenadas aproximadas
    grouped_scenes = {}
    logger.debug(f"Agrupando {len(all_item_data)} items...")
    
    for item in all_item_data:
        scene_id = item.get('id', '')
        
        # Extraer información clave para agrupamiento inteligente
        def extract_grouping_key(scene_id):
            # Para WorldView: extraer satélite + coordenadas aproximadas + fecha
            if re.match(r'WV\d{2}', scene_id):
                # Ejemplo: WV02N34_324027E062_3175002023101800000000MS00
                match = re.match(r'(WV\d{2})(N\d{2})_(\d{6})(E\d{3})_(\d{6})(\d{14})', scene_id)
                if match:
                    satellite, lat_deg, lat_min, lon_deg, lon_min, datetime_part = match.groups()
                    # Redondear coordenadas a un rango más amplio para agrupar escenas cercanas
                    lat_min_rounded = str(int(lat_min[:4]) // 10 * 10).zfill(4)  # Redondear a decenas
                    lon_min_rounded = str(int(lon_min[:4]) // 10 * 10).zfill(4)  # Redondear a decenas
                    date_part = datetime_part[:8]  # Solo fecha YYYYMMDD
                    return f"{satellite}{lat_deg}_{lat_min_rounded}{lon_deg}_{lon_min_rounded}_{date_part}"
            
            # Para otros satélites, usar nombre completo sin MS/PN
            base_name = scene_id
            if base_name.endswith('-calibrated'):
                base_name = base_name[:-11]
            base_name = re.sub(r'(MS|PN)\d{2}$', '', base_name)
            return base_name
        
        grouping_key = extract_grouping_key(scene_id)
        
        if grouping_key not in grouped_scenes:
            grouped_scenes[grouping_key] = {}
        
        if re.search(r"MS\d{2}", scene_id):
            grouped_scenes[grouping_key]['MS'] = item
            logger.debug(f"Agregada escena MS: {scene_id} -> {grouping_key}")
        elif re.search(r"PN\d{2}", scene_id):
            grouped_scenes[grouping_key]['PN'] = item
            logger.debug(f"Agregada escena PN: {scene_id} -> {grouping_key}")
        else:
            grouped_scenes[grouping_key]['OTRO'] = item
            logger.debug(f"Agregada escena OTRO: {scene_id} -> {grouping_key}")

    # Procesar escenas agrupadas
    final_items_to_process = []
    logger.debug(f"Procesando {len(grouped_scenes)} grupos de escenas...")
    for grouping_key, items in grouped_scenes.items():
        is_worldview = 'wv' in grouping_key.lower() or 'worldview' in grouping_key.lower()
        logger.debug(f"Grupo {grouping_key}: WorldView={is_worldview}, items={list(items.keys())}")
        if is_worldview:
            if 'MS' in items:
                final_items_to_process.append(items['MS'])
                logger.debug(f"Agregada escena WorldView MS: {grouping_key}")
            else:
                logger.warning(f"No se encontró escena MS para WorldView {grouping_key}")
        else:
            final_items_to_process.extend(items.values())
            logger.debug(f"Agregadas todas las escenas no-WorldView: {grouping_key}")

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
                'wv': ['wv', 'worldview', 'world view', 'wv02', 'wv03'],
                'ds': ['ds_phr', 'ds-phr', 'pleiades', 'pleiade', 'ds_phr1a', 'ds_phr1b'],
                'pleiades': ['pleiades', 'pleiade', 'ds_phr', 'ds-phr', 'ds_phr1a', 'ds_phr1b'],
                'bsg': ['bsg', 'beijing space view', 'bsg-116'],
                'k3a': ['k3a', 'kompsat', 'kompsat-3a', 'kompsat3a'],
                'gf2': ['gf2', 'gaofen', 'gaofen-2', 'gaofen2', 'gf2_pms2', 'gf2-pms2'],
                'geoeye': ['geoeye', 'geo eye', 'ge', 'geoeye-1', 'geoeye1'],
                'ge': ['geoeye', 'geo eye', 'ge', 'geoeye-1', 'geoeye1'],
                'all': ['wv', 'worldview', 'world view', 'wv02', 'wv03',
                        'ds_phr', 'ds-phr', 'pleiades', 'pleiade', 'ds_phr1a', 'ds_phr1b',
                        'bsg', 'beijing space view', 'bsg-116',
                        'k3a', 'kompsat', 'kompsat-3a', 'kompsat3a',
                        'gf2', 'gaofen', 'gaofen-2', 'gaofen2', 'gf2_pms2', 'gf2-pms2',
                        'geoeye', 'geo eye', 'ge', 'geoeye-1', 'geoeye1']
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
                        pixel_size_overview = abs(src.transform[0])
                        # Convertir unidades según el CRS (mismo que en read_gsd)
                        if src.crs:
                            crs_obj = CRS.from_string(str(src.crs))
                            if crs_obj.is_geographic:
                                scene_gsd = pixel_size_overview * 111320  # grados a metros
                            else:
                                scene_gsd = pixel_size_overview  # ya en metros
                        else:
                            scene_gsd = pixel_size_overview
                    
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
            band_files_found = list(calib_dir.glob("r-*.tif"))
            for band_file in band_files_found:
                band_name = band_file.stem.split('-')[-1]
                
                # Buscar versión _aligned primero
                aligned_band_file = band_file.parent / f"{band_file.stem}_aligned.tif"
                if aligned_band_file.exists():
                    bands[band_name] = aligned_band_file
                else:
                    bands[band_name] = band_file



            # -- Desactivar filtro de intervalo --
            # if interval_start and interval_end and scene_date:
            #     if scene_date < interval_start:
            #         continue
            #     
            #     if scene_date > interval_end + timedelta(hours=MAX_EXTENT_OVERLAP_HRS):
            #         logger.warning(f"{scene_id} fuera de intervalo "
            #                      f"({scene_date} > {interval_end})  — descartado")
            #         continue
            #     elif scene_date > interval_end:
            #         hours_over = (scene_date - interval_end).total_seconds() / 3600
            #         logger.info(f"{scene_id} {scene_date} a {hours_over:.1f} h "
            #                    f"del límite; lo acepto por margen +{MAX_EXTENT_OVERLAP_HRS} h")

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
            # logger.info(f"Escena añadida: {scene_id} con {len(bands)} bandas (GSD: {scene_gsd:.2f}m)")
            
        except Exception as e:
            logger.warning(f"Error procesando escena {item_data.get('id', 'unknown')}: {e}")
            continue
    
    # logger.info(f"Total de escenas procesadas: {len(scenes)}")
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
        # logger.info(f"Mejor escena PRE seleccionada: {pre_scene['scene_id']} "
        #            f"(fecha: {pre_scene['date']}, nubes: {pre_scene.get('cloud_cover', 'N/A')}%, "
        #            f"off-nadir: {pre_scene.get('sat:off_nadir', 'N/A')}°)")
        pass
    if post_scene:
        # logger.info(f"Mejor escena POST seleccionada: {post_scene['scene_id']} "
        #            f"(fecha: {post_scene['date']}, nubes: {post_scene.get('cloud_cover', 'N/A')}%, "
        #            f"off-nadir: {post_scene.get('sat:off_nadir', 'N/A')}°)")
        pass
    
    if not pre_scene and not post_scene:
        return None, None
    return pre_scene, post_scene

def buffer_geom(building, crs, buffer_meters):
    # Buffer consistente en metros
    if crs.to_epsg() != 3857:
        building_3857 = shapely.ops.transform(
            Transformer.from_crs(crs, 3857, always_xy=True).transform, building)
        buffered = building_3857.buffer(buffer_meters, cap_style=1)
        buffered = shapely.ops.transform(
            Transformer.from_crs(3857, crs, always_xy=True).transform, buffered)
    else:
        buffered = building.buffer(buffer_meters, cap_style=1)
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


TARGET_GSD = None  # Se determinará automáticamente basándose en las escenas


def sample_patches_for_percentiles_per_sensor(training_root, sample_size=1000):
    """Extrae parches aleatorios y calcula percentiles por banda y por sensor."""
    from random import shuffle
    sample_values = {}
    scenes = []
    # Buscar todas las escenas
    for event_dir in Path(training_root).glob('*'):
        for item_json in event_dir.glob("*-calibrated/*.json"):
            try:
                with open(item_json, 'r') as f:
                    item = json.load(f)
                sensor = item.get('properties', {}).get('platform', 'unknown').lower()
                calib_dir = item_json.parent / "Optical_Calibration"
                if not calib_dir.exists():
                    continue
                bands = {}
                for band_file in calib_dir.glob("r-*.tif"):
                    band_name = band_file.stem.split('-')[-1]
                    bands[band_name] = band_file
                if bands:
                    scenes.append((sensor, bands, calib_dir))
            except Exception:
                continue
    shuffle(scenes)
    count = {}
    for sensor, bands, calib_dir in scenes:
        for band_name, band_path in bands.items():
            try:
                with rasterio.open(band_path) as src:
                    h, w = src.height, src.width
                    if h < 64 or w < 64:
                        continue
                    y = np.random.randint(0, h-64)
                    x = np.random.randint(0, w-64)
                    patch = src.read(1, window=rasterio.windows.Window(x, y, 64, 64))
                    key = (sensor, band_name)
                    if key not in sample_values:
                        sample_values[key] = []
                        count[key] = 0
                    sample_values[key].append(patch.flatten())
                    count[key] += 1
                    if count[key] >= sample_size:
                        break
            except Exception:
                continue
    percentiles = {}
    for (sensor, band), vals_list in sample_values.items():
        vals = np.concatenate(vals_list)
        percentiles[(sensor, band)] = np.percentile(vals, [PERCENTILE_LOW, PERCENTILE_HIGH])
    return percentiles

def enhance_image_quality(img_array):
    """Mejora la calidad visual de la imagen aplicando ajustes automáticos de contraste y brillo."""
    if img_array is None:
        return None
    
    # Convertir a float32 si no lo está
    img = img_array.astype(np.float32)
    
    # Aplicar histogram stretching por canal
    enhanced = np.zeros_like(img)
    for i in range(img.shape[-1]):
        channel = img[..., i]
        
        # Calcular percentiles para stretching
        p1, p99 = np.percentile(channel, [1, 99])
        
        # Aplicar stretching
        if p99 > p1:
            channel_stretched = np.clip((channel - p1) / (p99 - p1), 0, 1)
        else:
            channel_stretched = channel
        
        # Ajuste de gamma sutil para mejorar contraste
        gamma = 0.85  # Más contraste
        channel_gamma = np.power(channel_stretched, gamma)
        
        enhanced[..., i] = channel_gamma
    
    return enhanced

def process_scene(scene, bbox, target_crs, is_pre, geom, tile_size):
    global P
    if scene is None:
        logger.warning(f"process_scene: scene es None para {'PRE' if is_pre else 'POST'}")
        return None
    if not scene["bands"]:
        logger.warning(f"process_scene: scene sin bandas para {'PRE' if is_pre else 'POST'} - scene_id: {scene.get('scene_id', 'Unknown')}")
        return None
    
    gsd = scene.get("gsd", 2.0)  # Valor por defecto si no hay GSD
    
    # Robustez: gsd debe ser finito y positivo
    try:
        gsd = float(gsd)
    except Exception:
        logger.warning(f"GSD no convertible a float: {gsd} en escena {scene['scene_id']}. Usando 2.0")
        gsd = 2.0
    
    if not np.isfinite(gsd) or gsd <= 0:
        logger.warning(f"GSD inválido ({gsd}) para escena {scene['scene_id']}, usando valor por defecto 2.0")
        gsd = 2.0
    
    # Detectar GSD extremadamente pequeño o grande que causa problemas
    if gsd < 0.1:
        logger.warning(f"GSD demasiado pequeño: {gsd} para escena {scene['scene_id']}, usando 2.0")
        gsd = 2.0
    elif gsd > 100:
        logger.warning(f"GSD demasiado grande: {gsd} para escena {scene['scene_id']}, usando 2.0")
        gsd = 2.0
    
    # Usar TARGET_GSD global si está disponible, sino usar un valor por defecto
    if TARGET_GSD is not None and np.isfinite(TARGET_GSD) and TARGET_GSD > 0:
        tgt_gsd = TARGET_GSD
    else:
        tgt_gsd = 2.0  # Valor por defecto si TARGET_GSD no está determinado
        logger.warning("TARGET_GSD no está determinado, usando valor por defecto 2.0m")
    
    # Mantener scale_orig, usar un flag no_upsample
    scale_orig = gsd / tgt_gsd
    if not np.isfinite(scale_orig) or scale_orig <= 0:
        logger.warning(f"Scale inválido ({scale_orig}) para escena {scene['scene_id']} (gsd={gsd}, target_gsd={tgt_gsd}), usando 1.0")
        scale_orig = 1.0
    
    # Validación adicional para scale muy pequeño que podría causar dimensiones 0
    if scale_orig < 0.001:  # Si el scale es menor a 0.001, es prácticamente 0
        logger.warning(f"Scale extremadamente pequeño detectado: {scale_orig:.10f} para escena {scene['scene_id']}, forzando scale = 1.0")
        scale_orig = 1.0
    
    # Determinar si no hacer upsample (preservar resolución nativa)
    # Para WorldView, ser más permisivo con el upsample para evitar imágenes muy pequeñas
    scene_id = scene.get('scene_id', '').lower()
    
    if 'worldview' in scene_id or 'wv' in scene_id:
        # WorldView: permitir upsample hasta 2x para evitar imágenes muy pequeñas
        no_upsample = NO_UPSAMPLE and scale_orig > 2.0
    else:
        # Otros sensores: comportamiento original
        no_upsample = NO_UPSAMPLE and scale_orig > 1.0
    
    scale = scale_orig  # ¡no tocar el scale original!
    
    band_arrays = []
    mask = None
    loaded_band_files = []  # Para mostrar los nombres de los archivos cargados
    
    # Cargar solo bandas RGB necesarias
    desired_order = ["red", "green", "blue"]
    bands_to_load = []
    
    # Buscar bandas RGB disponibles (priorizando versiones _aligned)
    for band in desired_order:
        # Primero buscar versión aligned, luego normal
        aligned_band = f"{band}_aligned"
        if aligned_band in scene["bands"]:
            bands_to_load.append(aligned_band)
        elif band in scene["bands"]:
            bands_to_load.append(band)
        else:
            # Si falta una banda, buscar sustitutos (también priorizando _aligned)
            if band == 'blue':
                if 'coastal_aligned' in scene["bands"]:
                    bands_to_load.append('coastal_aligned')
                elif 'coastal' in scene["bands"]:
                    bands_to_load.append('coastal')
                elif 'green_aligned' in scene["bands"]:
                    bands_to_load.append('green_aligned')
                elif 'green' in scene["bands"]:
                    bands_to_load.append('green')
                elif 'red_aligned' in scene["bands"]:
                    bands_to_load.append('red_aligned')
                elif 'red' in scene["bands"]:
                    bands_to_load.append('red')
                else:
                    # Usar la primera banda disponible
                    bands_to_load.append(list(scene["bands"].keys())[0])
            elif band == 'green':
                if 'red_aligned' in scene["bands"]:
                    bands_to_load.append('red_aligned')
                elif 'red' in scene["bands"]:
                    bands_to_load.append('red')
                else:
                    bands_to_load.append(list(scene["bands"].keys())[0])
            elif band == 'red':
                if 'green_aligned' in scene["bands"]:
                    bands_to_load.append('green_aligned')
                elif 'green' in scene["bands"]:
                    bands_to_load.append('green')
                else:
                    bands_to_load.append(list(scene["bands"].keys())[0])
            else:
                # Usar la primera banda disponible
                bands_to_load.append(list(scene["bands"].keys())[0])
    
    # Asegurar que tenemos al menos 3 bandas
    while len(bands_to_load) < 3:
        if list(scene["bands"].keys())[0] not in bands_to_load:
            bands_to_load.append(list(scene["bands"].keys())[0])
        else:
            # Si ya está la primera, usar la segunda
            bands_to_load.append(list(scene["bands"].keys())[1] if len(scene["bands"]) > 1 else list(scene["bands"].keys())[0])
    
    # Crear máscara una sola vez usando la primera banda
    first_band_path = scene["bands"][bands_to_load[0]]
    try:
        src = get_cached_dataset(first_band_path)
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
            logger.warning(f"Ventana vacía para escena {scene['scene_id']}, banda {bands_to_load[0]}: {win}")
            return None
        
        # Crear máscara una sola vez
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
            logger.warning(f"Máscara vacía para escena {scene['scene_id']}")
            return None

        # Re-muestrear la máscara directamente al tamaño objetivo
        if no_upsample:
            # Sin remuestreo, solo redimensionar si es necesario
            if mask_full.shape != (tile_size, tile_size):
                mask = cv2.resize(
                    mask_full.astype(np.uint8),
                    (tile_size, tile_size),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
            else:
                mask = mask_full.astype(np.uint8)
        else:
            # Remuestrear la máscara con el mismo método que los datos
            mask = cv2.resize(
                mask_full.astype(np.uint8),
                (tile_size, tile_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
            
    except Exception as e:
        logger.warning(f"Error creando máscara para escena {scene['scene_id']}: {e}")
        return None
    
    # Procesar todas las bandas usando el cache
    for band_name in bands_to_load:
        band_path = scene["bands"][band_name]
        try:
            src = get_cached_dataset(band_path)
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
            
            # === NUEVA LÓGICA: DIVIDIR FLUJOS CON FLAG no_upsample ===
            if no_upsample:
                # Leer a resolución nativa y centrar en un lienzo 224×224
                data = src.read(
                    1, window=win, boundless=True, fill_value=0,
                    resampling=Resampling.nearest
                )
                # Crear canvas del tamaño objetivo y centrar los datos nativos
                canvas = np.zeros((tile_size, tile_size), data.dtype)
                h, w = data.shape
                y0 = (tile_size - h) // 2
                x0 = (tile_size - w) // 2
                
                # Asegurar que no se excedan los límites del canvas
                y1 = min(y0 + h, tile_size)
                x1 = min(x0 + w, tile_size)
                data_h = y1 - y0
                data_w = x1 - x0
                
                if y0 >= 0 and x0 >= 0:
                    canvas[y0:y1, x0:x1] = data[:data_h, :data_w]
                else:
                    # Si los datos nativos son más grandes que el canvas, centrar tomando la parte central
                    src_y0 = max(0, -y0)
                    src_x0 = max(0, -x0)
                    dst_y0 = max(0, y0)
                    dst_x0 = max(0, x0)
                    copy_h = min(h - src_y0, tile_size - dst_y0)
                    copy_w = min(w - src_x0, tile_size - dst_x0)
                    canvas[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = data[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
                
                data = canvas
            else:
                # Comportamiento normal con remuestreo directo
                resampling = (Resampling.average   if scale < 1 else
                             Resampling.bilinear  if scale > 1 else
                             Resampling.nearest
                             )
                
                # Leer directamente al tamaño objetivo con un solo remuestreo
                data = src.read(
                    1, 
                    window=win, 
                    boundless=True, 
                    fill_value=0,
                    out_shape=(tile_size, tile_size),
                    resampling=resampling
                )
            
            if data.size == 0:
                logger.warning(f"Datos vacíos para banda {band_name} en escena {scene['scene_id']}")
                continue

            band_arrays.append(data)
            loaded_band_files.append((band_name, str(band_path)))
        except Exception as e:
            logger.warning(f"Error procesando banda {band_name} en escena {scene['scene_id']}: {e}")
            continue
    if not band_arrays:
        logger.warning(f"Todas las bandas fallaron para escena {'PRE' if is_pre else 'POST'} {scene['scene_id']}")
        return None
    
    # Actualizar band_names para reflejar solo las bandas cargadas exitosamente
    band_names = [name for name, _ in loaded_band_files]
    
    # Si no se creó máscara (por problemas con geometrías), crear una máscara por defecto
    if mask is None:
        logger.warning(f"No se pudo crear máscara para escena {scene['scene_id']}, usando máscara por defecto")
        mask = np.ones((tile_size, tile_size), dtype=np.uint8)
    
    # === NORMALIZACIÓN POR PERCENTILES ===
    full_stack = np.stack(band_arrays, axis=-1).astype(np.float32)
    # Si más del 50% del stack es 0, devolver None
    zero_frac = np.mean(full_stack == 0)
    
    # Sólo descarta si *sí* había upsample y más de la mitad son ceros
    # Cuando no hay upsample el lienzo puede llevar mucho relleno 0 → relájalo
    if not no_upsample and zero_frac > 0.5:
        return None
    
    # Usar variable global P que ahora está disponible en los workers
    global P
    sensor = scene.get('properties', {}).get('platform', 'unknown').lower() if 'properties' in scene else scene.get('sensor', 'unknown').lower()
    for i, band_name in enumerate(band_names):
        pkey = (sensor, band_name)
        if P and pkey in P:
            p1, p99 = P[pkey]  # Usar PERCENTILE_LOW/HIGH (1-99%)
            full_stack[..., i] = np.clip(full_stack[..., i], p1, p99)
            full_stack[..., i] = (full_stack[..., i] - p1) / max(p99 - p1, 1e-8)
        else:
            mn = full_stack[..., i].min()
            mx = full_stack[..., i].max()
            full_stack[..., i] = (full_stack[..., i] - mn) / max(mx - mn, 1e-8)
    
    # Aplicar mejora de calidad visual
    full_stack = enhance_image_quality(full_stack)

    # Reordenar a RGB
    desired_order = ["red", "green", "blue"]
    ordered_arrays = []
    
    for band in desired_order:
        # Primero buscar versión aligned, luego normal
        aligned_band = f"{band}_aligned"
        if aligned_band in band_names:
            idx = band_names.index(aligned_band)
            ordered_arrays.append(full_stack[..., idx])
        elif band in band_names:
            idx = band_names.index(band)
            ordered_arrays.append(full_stack[..., idx])
        else:
            # Si falta la banda, usar una banda sustituta apropiada (priorizando _aligned)
            if band == 'blue':
                # Para banda azul, priorizar coastal (WorldView-2), luego green, luego red
                if 'coastal_aligned' in band_names:
                    substitute_band = 'coastal_aligned'
                elif 'coastal' in band_names:
                    substitute_band = 'coastal'
                elif 'green_aligned' in band_names:
                    substitute_band = 'green_aligned'
                elif 'green' in band_names:
                    substitute_band = 'green'
                elif 'red_aligned' in band_names:
                    substitute_band = 'red_aligned'
                elif 'red' in band_names:
                    substitute_band = 'red'
                else:
                    substitute_band = band_names[0]
            elif band == 'green':
                # Para banda verde, usar rojo como sustituto
                if 'red_aligned' in band_names:
                    substitute_band = 'red_aligned'
                elif 'red' in band_names:
                    substitute_band = 'red'
                else:
                    substitute_band = band_names[0]
            elif band == 'red':
                # Para banda roja, usar verde como sustituto
                if 'green_aligned' in band_names:
                    substitute_band = 'green_aligned'
                elif 'green' in band_names:
                    substitute_band = 'green'
                else:
                    substitute_band = band_names[0]
            else:
                # Usar la primera banda disponible
                substitute_band = band_names[0]
            
            substitute_idx = band_names.index(substitute_band)
            ordered_arrays.append(full_stack[..., substitute_idx])
    
    # Si hay menos de 3 bandas, rellena hasta 3
    while len(ordered_arrays) < 3:
        ordered_arrays.append(np.zeros_like(full_stack[..., 0]))

    rgb = np.stack(ordered_arrays, axis=-1)



    # Apilar máscara después de RGB (solo RGB + máscara = 4 canales)
    stack_final = np.concatenate([rgb, mask[..., None]], axis=-1)

    try:
        result = (stack_final.transpose(2, 0, 1) * 65535).round().astype(np.uint16)
        return result
    except Exception as e:
        logger.warning(f"Error en normalización final para escena {scene['scene_id']}: {e}")
        return None





def _init_worker(shared_P, shared_scenes, tgt_gsd):
    """Inicializador para workers del ProcessPoolExecutor."""
    global P, SCENES, TARGET_GSD, RASTER_CACHE
    P = shared_P         # diccionario de percentiles
    SCENES = shared_scenes
    TARGET_GSD = tgt_gsd
    RASTER_CACHE = {}    # Cache para datasets de rasterio
    # Log para confirmar que los percentiles se pasaron correctamente
    if P:
        logger.debug(f"Worker inicializado con {len(P)} percentiles y TARGET_GSD={TARGET_GSD}")
    else:
        logger.warning("Worker inicializado sin percentiles (P=None)")

def get_cached_dataset(path):
    """Obtiene un dataset de rasterio con cache para evitar aperturas repetidas."""
    global RASTER_CACHE
    if path not in RASTER_CACHE:
        RASTER_CACHE[path] = rasterio.open(path, sharing=True)
    return RASTER_CACHE[path]

def cleanup_raster_cache():
    """Limpia el cache de datasets de rasterio para liberar memoria."""
    global RASTER_CACHE
    for dataset in RASTER_CACHE.values():
        try:
            dataset.close()
        except:
            pass
    RASTER_CACHE.clear()

def process_tile_worker(args):
    wkb_geom, scenes_data, crs_wkt, output_dir, fid, damage, buffer_meters, tile_size, use_multi_scene = args
    geometry = wkb.loads(wkb_geom)
    crs = CRS.from_wkt(crs_wkt)
    
    # Extraer scenes, activation_date y target_gsd del tuple
    if len(scenes_data) == 3:
        scenes, activation_date, target_gsd = scenes_data
    else:
        # Backward compatibility
        scenes, activation_date = scenes_data
        target_gsd = 2.0
    
    # Configurar TARGET_GSD para este worker
    global TARGET_GSD
    TARGET_GSD = target_gsd
    
    try:
        if use_multi_scene:
            return process_tile_multi_scene(geometry, (scenes, activation_date), crs, output_dir, fid, damage, buffer_meters, tile_size)
        else:
            return process_tile(geometry, (scenes, activation_date), crs, output_dir, fid, damage, buffer_meters, tile_size)
    except Exception as e:
        # Solo logear errores críticos que no deberían ocurrir
        logger.error(f"Error crítico procesando FID {fid}: {e}")
        return None


def pick_scene_for_geom_with_fallback(scenes, geom, crs, is_pre, act_dt, buffer_meters):
    """
    Selecciona la mejor escena para un polígono usando un enfoque de fallback progresivo.
    
    Estrategia:
    1. Primero busca escenas con alta cobertura (>=80%)
    2. Si no encuentra, reduce progresivamente los requisitos: 60%, 40%, 20%
    3. Como último recurso, usa cualquier escena que tenga intersección
    4. Prioriza por cobertura, luego por cercanía temporal
    
    Args:
        scenes: Lista de escenas disponibles
        geom: Geometría del polígono/edificio
        crs: Sistema de coordenadas
        is_pre: True para escenas PRE, False para POST
        act_dt: Fecha de activación del terremoto
        buffer_meters: Buffer en metros alrededor del polígono
    
    Returns:
        Escena seleccionada o None si no hay ninguna del período correcto
    """
    buffered_geom = buffer_geom(geom, crs, buffer_meters)
    
    # Transform buffered geometry to WGS84 (EPSG:4326) to match scene geometries
    if crs.to_epsg() != 4326:
        try:
            transformer_to_wgs = Transformer.from_crs(crs, 4326, always_xy=True)
            buffered_geom_wgs = shapely.ops.transform(transformer_to_wgs.transform, buffered_geom)
        except Exception as e:
            logger.warning(f"Error transforming buffered geometry to WGS84: {e}")
            buffered_geom_wgs = buffered_geom  # fallback – may lead to 0-coverage
    else:
        buffered_geom_wgs = buffered_geom
    
    # Filtrar escenas por período temporal primero
    temporal_candidates = []
    for scene in scenes:
        if not scene["date"]:
            continue

        # Asegurarse que la fecha de la escena tiene timezone
        if scene["date"].tzinfo is None:
            scene["date"] = scene["date"].replace(tzinfo=timezone.utc)

        # Comprobar si la escena es pre o post
        is_candidate = (is_pre and scene["date"] <= act_dt) or (
            not is_pre and scene["date"] >= act_dt
        )
        if not is_candidate:
            continue

        # Umbral temporal para escenas pre-evento – ¡eliminar!
        # if is_pre and (act_dt - scene["date"]).days > 550:
        #     continue
            
        temporal_candidates.append(scene)
    
    if not temporal_candidates:
        logger.debug(f"No hay escenas {'PRE' if is_pre else 'POST'} disponibles para el período")
        return None
    
    logger.debug(f"Encontradas {len(temporal_candidates)} escenas {'PRE' if is_pre else 'POST'} en el período correcto")
    
    # Niveles de cobertura mínima progresivos
    coverage_thresholds = [0.8, 0.6, 0.4, 0.2, 0.05, 0.01]  # 80%, 60%, 40%, 20%, 5%, 1%
    
    for min_cover in coverage_thresholds:
        candidates = []
        
        # Evaluar cobertura para cada escena
        for scene in temporal_candidates:
            try:
                # Use WGS84-aligned geometry for intersection
                intersection = scene["geometry"].intersection(buffered_geom_wgs)
                cover = intersection.area / buffered_geom_wgs.area if buffered_geom_wgs.area > 0 else 0
            except Exception as e:
                logger.warning(f"Error calculando intersección para escena {scene['scene_id']}: {e}")
                cover = 0

            if cover >= min_cover:
                candidates.append((scene, cover))

        if candidates:
            # Ordenar por cobertura (descendente) y luego por cercanía temporal (ascendente)
            def sort_key(t):
                scene, cover = t
                temporal_dist = abs(act_dt - scene["date"]).total_seconds()
                cloud_cover = scene.get("cloud_cover", 999)
                off_nadir = scene.get("sat:off_nadir", 999)
                
                # Priorizar cobertura alta, luego cercanía temporal
                return (-cover, temporal_dist, cloud_cover, off_nadir)
            
            candidates.sort(key=sort_key)
            best_scene, best_cover = candidates[0]
            
            logger.debug(f"Escena {'PRE' if is_pre else 'POST'} seleccionada: {best_scene['scene_id']} "
                        f"(cobertura: {best_cover:.2f}, umbral: {min_cover:.2f}, "
                        f"fecha: {best_scene['date'].date()})")
            
            return best_scene
            
        logger.debug(f"No se encontraron escenas con cobertura >= {min_cover:.1%}")
    
    # Si llegamos aquí, no hay ninguna escena que cubra el polígono ni siquiera mínimamente
    logger.warning(f"No se encontró escena {'PRE' if is_pre else 'POST'} que cubra el polígono "
                  f"(probado hasta {coverage_thresholds[-1]:.1%} de cobertura). "
                  f"Esto puede indicar que el edificio está fuera del área de cobertura de las escenas disponibles.")
    return None




def find_best_scene_for_polygon(scenes, geom, crs, is_pre, act_dt, buffer_meters, min_cover=0.6):
    """
    Encuentra la mejor escena que cubra un polígono específico.
    
    Args:
        scenes: Lista de todas las escenas disponibles
        geom: Geometría del polígono
        crs: Sistema de coordenadas
        is_pre: True para escenas PRE, False para POST
        act_dt: Fecha de activación del terremoto
        buffer_meters: Buffer alrededor del polígono
        min_cover: Cobertura mínima requerida (default 60%)
    
    Returns:
        Mejor escena para este polígono o None
    """
    buffered_geom = buffer_geom(geom, crs, buffer_meters)
    # Transform to WGS84 to align with scene geometries
    if crs.to_epsg() != 4326:
        try:
            transformer_to_wgs = Transformer.from_crs(crs, 4326, always_xy=True)
            buffered_geom_wgs = shapely.ops.transform(transformer_to_wgs.transform, buffered_geom)
        except Exception as e:
            logger.warning(f"Error transforming buffered geometry to WGS84: {e}")
            buffered_geom_wgs = buffered_geom
    else:
        buffered_geom_wgs = buffered_geom
    
    candidates = []
    
    for scene in scenes:
        if not scene["date"]:
            continue

        # Asegurar timezone
        if scene["date"].tzinfo is None:
            scene["date"] = scene["date"].replace(tzinfo=timezone.utc)

        # Filtrar por período temporal
        is_candidate = (is_pre and scene["date"] <= act_dt) or (
            not is_pre and scene["date"] >= act_dt
        )
        if not is_candidate:
            continue

        # Umbral temporal para pre-evento – ¡eliminar!
        # if is_pre and (act_dt - scene["date"]).days > 550:
        #     continue

        # Calcular cobertura
        try:
            # Use WGS84-aligned geometry for intersection
            intersection = scene["geometry"].intersection(buffered_geom_wgs)
            cover = intersection.area / buffered_geom_wgs.area if buffered_geom_wgs.area > 0 else 0
        except Exception as e:
            logger.warning(f"Error calculando intersección para escena {scene['scene_id']}: {e}")
            cover = 0

        if cover >= min_cover:
            candidates.append((scene, cover))

    if not candidates:
        return None

    # Ordenar por cobertura (desc) y luego cercanía temporal (asc)
    def sort_key(t):
        scene, cover = t
        temporal_dist = abs(act_dt - scene["date"]).total_seconds()
        cloud_cover = scene.get("cloud_cover", 999)
        off_nadir = scene.get("sat:off_nadir", 999)
        return (-cover, temporal_dist, cloud_cover, off_nadir)
    
    candidates.sort(key=sort_key)
    return candidates[0][0]


def create_scene_polygon_mapping(all_scenes, polygons, crs, activation_date, buffer_meters):
    """
    Crea un mapeo de qué escenas usar para cada polígono para maximizar la cobertura.
    
    Args:
        all_scenes: Lista de todas las escenas disponibles
        polygons: Lista de polígonos a procesar
        crs: Sistema de coordenadas
        activation_date: Fecha de activación del terremoto
        buffer_meters: Buffer alrededor de polígonos
    
    Returns:
        dict: {
            'pre_mapping': {polygon_idx: scene},
            'post_mapping': {polygon_idx: scene},
            'coverage_stats': {...}
        }
    """
    pre_mapping = {}
    post_mapping = {}
    coverage_stats = {
        'total_polygons': len(polygons),
        'pre_covered': 0,
        'post_covered': 0,
        'pre_scenes_used': set(),
        'post_scenes_used': set()
    }
    
    logger.info(f"Creando mapeo de escenas para {len(polygons)} polígonos...")
    
    for idx, polygon in enumerate(polygons):
        # Buscar mejor escena PRE para este polígono
        pre_scene = find_best_scene_for_polygon(
            all_scenes, polygon, crs, is_pre=True, 
            act_dt=activation_date, buffer_meters=buffer_meters
        )
        
        # Buscar mejor escena POST para este polígono  
        post_scene = find_best_scene_for_polygon(
            all_scenes, polygon, crs, is_pre=False, 
            act_dt=activation_date, buffer_meters=buffer_meters
        )
        
        if pre_scene:
            pre_mapping[idx] = pre_scene
            coverage_stats['pre_covered'] += 1
            coverage_stats['pre_scenes_used'].add(pre_scene['scene_id'])
            
        if post_scene:
            post_mapping[idx] = post_scene
            coverage_stats['post_covered'] += 1
            coverage_stats['post_scenes_used'].add(post_scene['scene_id'])
            
        if idx % 50 == 0:  # Log progreso cada 50 polígonos
            logger.debug(f"Procesados {idx}/{len(polygons)} polígonos...")
    
    # Estadísticas finales
    pre_coverage_pct = (coverage_stats['pre_covered'] / coverage_stats['total_polygons']) * 100
    post_coverage_pct = (coverage_stats['post_covered'] / coverage_stats['total_polygons']) * 100
    
    logger.info(f"Mapeo completado:")
    logger.info(f"  PRE: {coverage_stats['pre_covered']}/{coverage_stats['total_polygons']} polígonos cubiertos ({pre_coverage_pct:.1f}%)")
    logger.info(f"  POST: {coverage_stats['post_covered']}/{coverage_stats['total_polygons']} polígonos cubiertos ({post_coverage_pct:.1f}%)")
    logger.info(f"  Escenas PRE usadas: {len(coverage_stats['pre_scenes_used'])}")
    logger.info(f"  Escenas POST usadas: {len(coverage_stats['post_scenes_used'])}")
    
    return {
        'pre_mapping': pre_mapping,
        'post_mapping': post_mapping,
        'coverage_stats': coverage_stats
    }







def process_tile_multi_scene(building, scenes, crs, output_dir, fid, damage_label, buffer_meters, tile_size=224):
    """
    Versión mejorada de process_tile que puede usar múltiples escenas por tile para maximizar la cobertura de polígonos.
    
    En lugar de seleccionar 1 escena PRE + 1 escena POST para todo el tile,
    encuentra la mejor escena PRE y POST para cada polígono individual.
    """
    all_scenes, activation_date = scenes
    
    # Usar método directo sin crear mapeo masivo para un solo polígono
    pre_scene = pick_scene_for_geom_with_fallback(
        all_scenes, building, crs, is_pre=True, act_dt=activation_date, buffer_meters=buffer_meters
    )
    post_scene = pick_scene_for_geom_with_fallback(
        all_scenes, building, crs, is_pre=False, act_dt=activation_date, buffer_meters=buffer_meters
    )
    
    if not pre_scene and not post_scene:
        # logger.warning(f"No se encontraron escenas PRE ni POST para el polígono FID {fid}")
        return None
        
    if not pre_scene:
        # logger.warning(f"No se encontró escena PRE para el polígono FID {fid}")
        # Fallback a selección global
        pre_scene_global, _ = select_best_scenes(all_scenes, activation_date)
        pre_scene = pre_scene_global
        
    if not post_scene:
        # logger.warning(f"No se encontró escena POST para el polígono FID {fid}")
        # Fallback a selección global
        _, post_scene_global = select_best_scenes(all_scenes, activation_date)
        post_scene = post_scene_global

    # Log de las escenas seleccionadas para este polígono (solo en debug)
    if pre_scene and post_scene:
        pass  # Log removido para limpiar output
    
    # Resto del procesamiento igual que process_tile original
    if pre_scene and post_scene:
        # Validación temporal
        if not (pre_scene["date"] < activation_date <= post_scene["date"]):
            logger.error("¡Incoherencia temporal! Abort tile.")
            return None

    # Buffer consistente
    buffered_geom = buffer_geom(building, crs, buffer_meters)
    if not buffered_geom.is_valid:
        buffered_geom = buffered_geom.buffer(0)
    bbox = buffered_geom.bounds
    
    # Procesar escena PRE con fallback sistemático
    pre_data = None
    if pre_scene:
        pre_data = process_scene(pre_scene, bbox, crs, is_pre=True, geom=building, tile_size=tile_size)
        
        # Si falló, probar todas las escenas disponibles
        if pre_data is None:
            # Obtener todas las escenas PRE ordenadas por prioridad
            pre_scenes = [s for s in all_scenes if s["date"] < activation_date]
            if pre_scenes:
                # Ordenar por fecha (más cercana a la activación primero)
                pre_scenes.sort(key=lambda x: abs((activation_date - x["date"]).total_seconds()))
                
                for i, alt_scene in enumerate(pre_scenes[:5]):  # Probar máximo 5 escenas
                    if alt_scene['scene_id'] == pre_scene['scene_id']:
                        continue  # Saltar la escena que ya falló
                        
                    alt_data = process_scene(alt_scene, bbox, crs, is_pre=True, geom=building, tile_size=tile_size)
                    
                    if alt_data is not None:
                        pre_scene = alt_scene  # Actualizar la escena usada
                        pre_data = alt_data
                        break
    
    # Procesar escena POST con fallback sistemático
    post_data = None
    if post_scene:
        post_data = process_scene(post_scene, bbox, crs, is_pre=False, geom=building, tile_size=tile_size)
        
        # Si falló, probar todas las escenas disponibles
        if post_data is None:
            # Obtener todas las escenas POST ordenadas por prioridad
            post_scenes = [s for s in all_scenes if s["date"] >= activation_date]
            if post_scenes:
                # Ordenar por fecha (más cercana a la activación primero)
                post_scenes.sort(key=lambda x: abs((activation_date - x["date"]).total_seconds()))
                
                for i, alt_scene in enumerate(post_scenes[:5]):  # Probar máximo 5 escenas
                    if alt_scene['scene_id'] == post_scene['scene_id']:
                        continue  # Saltar la escena que ya falló
                        
                    alt_data = process_scene(alt_scene, bbox, crs, is_pre=False, geom=building, tile_size=tile_size)
                    
                    if alt_data is not None:
                        post_scene = alt_scene  # Actualizar la escena usada
                        post_data = alt_data
                        break

    # Sin alineación entre escenas pre/post
    shift_xy = (0.0, 0.0)

    if pre_data is None:
        pre_data = np.zeros((1, tile_size, tile_size), np.uint16)
        has_pre = 0
    else:
        has_pre = 1
    if post_data is None:
        post_data = np.zeros((1, tile_size, tile_size), np.uint16)
        has_post = 0
    else:
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


# Mantener la función original como fallback
def process_tile(building, scenes, crs, output_dir, fid, damage_label, buffer_meters, tile_size=224):
    all_scenes, activation_date = scenes
    # logger.debug(f"All scenes before select_best_scenes: {all_scenes}")
    pre_scene_global, post_scene_global = select_best_scenes(all_scenes, activation_date)
    pre_scene = (
        pick_scene_for_geom_with_fallback(all_scenes, building, crs, True, activation_date, buffer_meters)
        or pre_scene_global
    )
    post_scene = (
        pick_scene_for_geom_with_fallback(all_scenes, building, crs, False, activation_date, buffer_meters)
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
    buffered_geom = buffer_geom(building, crs, buffer_meters)
    # Sanity-check para evitar geometrías self-intersecting
    if not buffered_geom.is_valid:
        buffered_geom = buffered_geom.buffer(0)
    bbox = buffered_geom.bounds
    pre_data = (
        process_scene(pre_scene, bbox, crs, is_pre=True, geom=building, tile_size=tile_size)
        if pre_scene
        else None
    )
    post_data = (
        process_scene(post_scene, bbox, crs, is_pre=False, geom=building, tile_size=tile_size)
        if post_scene
        else None
    )

    # Sin alineación entre escenas pre/post
    shift_xy = (0.0, 0.0)

    if pre_data is None:
        pre_data = np.zeros((1, tile_size, tile_size), np.uint16)
        has_pre = 0
        logger.warning(f"FID {fid}: No se encontró escena PRE disponible")
    else:
        has_pre = 1
    if post_data is None:
        post_data = np.zeros((1, tile_size, tile_size), np.uint16)
        has_post = 0
        logger.warning(f"FID {fid}: No se encontró escena POST disponible")
    else:
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

def process_gpkg(gpkg_path, training_root, output_root, split, buffer_meters, max_workers=12, satellites=None, tile_size=224, use_multi_scene=True, target_gsd=None):
    global TARGET_GSD
    # Extraer nombre único de área/ciudad del archivo GPKG
    city_area = gpkg_path.stem.replace('_Training', '').replace('_training', '').replace('_Testing', '').replace('_testing', '').lower()
    output_dir = output_root / split / city_area
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = load_gpkg_with_fiona(gpkg_path)
    # logger.info(f"Procesando {gpkg_path.name} ({len(gdf)} edificios)")
    # logger.info(f"Edificios en GPKG: {len(gdf)}")
    
    # Print columns for debugging Afghanistan areas
    if "afghanistan" in city_area:
        logger.debug(f"Columnas en {gpkg_path}: {list(gdf.columns)}")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    gdf['geometry'] = gdf['geometry'].buffer(0)
    # logger.info(f"Edificios tras limpieza: {len(gdf)}")
    
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

    # logger.info(f"Columna FID: {fid_col}")
    event_name = gpkg_path.parent.name
    call_dir = training_root / event_name
    activation_date = get_activation_date(call_dir)
    # logger.info(f"Fecha de activación: {activation_date}")
    
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
    scenes = find_scenes_for_aoi(aoi_geom_wgs, call_dir, None, None, satellites)
    logger.info(f"Escenas encontradas: {len(scenes)}")
    if scenes:
        for i, scene in enumerate(scenes, 1):
            logger.info(f"  {i}. {scene['scene_id']} (GSD: {scene['gsd']:.2f}m, fecha: {scene['date'].strftime('%Y-%m-%d') if scene['date'] else 'N/A'})")
    
    if not scenes:
        logger.warning(f"No se encontraron escenas para {gpkg_path}")
        return []
    
    # Determinar TARGET_GSD automáticamente basándose en las escenas o usar el especificado por el usuario
    if target_gsd is not None:
        TARGET_GSD = target_gsd
        # logger.info(f"Usando TARGET_GSD especificado por el usuario: {TARGET_GSD:.2f}m")
    else:
        # logger.info("Determinando TARGET_GSD automáticamente basándose en las escenas...")
        TARGET_GSD = determine_optimal_target_gsd(scenes)
        # logger.info(f"TARGET_GSD determinado automáticamente: {TARGET_GSD:.2f}m")
    
    # Seleccionar las mejores escenas globales para mostrar fechas de referencia
    pre_scene_global, post_scene_global = select_best_scenes(scenes, activation_date)
    # if pre_scene_global:
    #     logger.info(f"Mejor escena PRE seleccionada: {pre_scene_global['scene_id']} "
    #                f"(fecha: {pre_scene_global['date']}, nubes: {pre_scene_global.get('cloud_cover', 'None')}%, "
    #                f"off-nadir: {pre_scene_global.get('sat:off_nadir', 'None')}°)")
    # else:
    #     logger.info("No se encontró escena PRE de referencia")
    
    # if post_scene_global:
    #     logger.info(f"Mejor escena POST seleccionada: {post_scene_global['scene_id']} "
    #                f"(fecha: {post_scene_global['date']}, nubes: {post_scene_global.get('cloud_cover', 'None')}%, "
    #                f"off-nadir: {post_scene_global.get('sat:off_nadir', 'None')}°)")
    # else:
    #     logger.info("No se encontró escena POST de referencia")
    
    meta_data = []
    damage_col = 'damaged' if 'damaged' in gdf.columns else None
    # logger.info(f"Columna de daño: {damage_col}")
    
    crs_wkt = gdf.crs.to_wkt()
    tasks = []
    for idx, row in gdf.iterrows():
        fid = row[fid_col]
        damage = row[damage_col] if damage_col else -1
        
        # Pasamos todas las escenas, la fecha de activación y TARGET_GSD a cada worker
        task = (row.geometry.wkb, (scenes, activation_date, TARGET_GSD), crs_wkt, output_dir, fid, damage, buffer_meters, tile_size, use_multi_scene)
        tasks.append(task)

    logger.debug(f"Preparados {len(tasks)} tasks para procesamiento")
    
    # Log para confirmar que los percentiles están disponibles
    # if P:
    #     logger.info(f"Pasando {len(P)} percentiles a {max_workers} workers")
    # else:
    #     logger.warning("No hay percentiles disponibles para pasar a los workers")

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

        # Usar initializer para pasar variables globales a los workers
        with ProcessPoolExecutor(max_workers=max_workers,
                                initializer=_init_worker,
                                initargs=(P, scenes, TARGET_GSD)) as executor:
            try:
                processed_count = 0
                failed_count = 0
                
                # Usar submit + as_completed para mejor rendimiento
                futures = [executor.submit(process_tile_worker, task) for task in tasks]
                
                # Procesar futures con barra de progreso para edificios del archivo actual
                with tqdm(total=len(tasks), desc=f"Edificios {gpkg_path.name}", leave=True) as pbar:
                    for future in as_completed(futures):
                        try:
                            meta = future.result()
                            if meta:
                                meta['gpkg'] = gpkg_path.name
                                meta['event'] = event_name
                                meta['split'] = split
                                meta_data.append(meta)
                                processed_count += 1
                            else:
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"Error en worker: {e}")
                            failed_count += 1
                        
                        pbar.update(1)
                
                # Mostrar resumen al final
                if failed_count > 0:
                    logger.warning(f"Resumen {gpkg_path.name}: {processed_count} exitosos, {failed_count} fallidos")
                else:
                    logger.info(f"Resumen {gpkg_path.name}: {processed_count} tiles procesados exitosamente")
                
                # Contar tiles con problemas de escenas
                tiles_without_pre = sum(1 for meta in meta_data if meta.get('has_pre', 1) == 0)
                tiles_without_post = sum(1 for meta in meta_data if meta.get('has_post', 1) == 0)
                tiles_without_both = sum(1 for meta in meta_data if meta.get('has_pre', 1) == 0 and meta.get('has_post', 1) == 0)
                
                if tiles_without_pre > 0 or tiles_without_post > 0:
                    logger.warning(f"⚠️  Tiles con problemas de cobertura:")
                    logger.warning(f"   - Sin escena PRE: {tiles_without_pre}")
                    logger.warning(f"   - Sin escena POST: {tiles_without_post}")
                    logger.warning(f"   - Sin ambas escenas: {tiles_without_both}")
                    logger.warning(f"   - Porcentaje con problemas: {((tiles_without_pre + tiles_without_post - tiles_without_both) / len(meta_data) * 100):.1f}%")
                    
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
    parser.add_argument('--buffer-meters', type=int, default=20, 
                       help='Buffer en metros alrededor de cada edificio')
    parser.add_argument('--max-workers', type=int, default=12, 
                       help='Número máximo de workers para procesamiento paralelo')
    parser.add_argument('--satellites', type=str, default='all', 
                       help='Satélites a usar (separados por coma). Opciones: ' +
                            'wv (WorldView-2/3), ds/pleiades (Pleiades), bsg (Beijing Space View), ' +
                            'k3a (Kompsat-3A), gf2 (Gaofen-2), geoeye/ge (GeoEye-1), all (todos). ' +
                            'Por defecto: wv,ds')

    parser.add_argument('--country', type=str, default=None, 
                       help='Filtrar y procesar solo escenas de este país. Opciones: Turkey, Myanmar, China, Syria, Morocco, Afghanistan')
    parser.add_argument('--use-multi-scene', action='store_true', default=True,
                       help='Usar sistema multi-escena para maximizar cobertura de polígonos. ' +
                            'Si es True, cada polígono se procesa con la mejor escena que lo cubra. ' +
                            'Si es False, usa el método tradicional de 1 escena PRE + 1 escena POST por tile.')
    
    parser.add_argument('--target-gsd', type=float, default=None,
                       help='GSD objetivo en metros para el remuestreo. Si no se especifica, ' +
                            'se determina automáticamente basándose en las escenas disponibles. ' +
                            'Valores típicos: 0.3-0.5m para WorldView, 1.0m para Pleiades, 2.0m para otros.')

    args = parser.parse_args()
    # Configurar nivel de logging según argumento
    log_level = getattr(logging, args.loglevel.upper())
    logger.setLevel(log_level)
    
    # Validar satélites seleccionados
    satellites_list = [s.strip().lower() for s in args.satellites.split(',')]
    valid_satellites = ['wv', 'ds', 'pleiades', 'bsg', 'k3a', 'gf2', 'geoeye', 'ge', 'all']
    invalid_satellites = [s for s in satellites_list if s not in valid_satellites]
    
    if invalid_satellites:
        logger.error(f"Satélites inválidos especificados: {invalid_satellites}")
        logger.error(f"Satélites válidos: {', '.join(valid_satellites)}")
        return
    
    # Mostrar información detallada sobre los satélites seleccionados
    satellite_info = {
        'wv': 'WorldView-2/3 (WV02, WV03)',
        'ds': 'Pleiades (DS_PHR1A, DS_PHR1B)', 
        'pleiades': 'Pleiades (DS_PHR1A, DS_PHR1B)',
        'bsg': 'Beijing Space View (BSG-116)',
        'k3a': 'Kompsat-3A',
        'gf2': 'Gaofen-2 (GF2_PMS2)',
        'geoeye': 'GeoEye-1',
        'ge': 'GeoEye-1',
        'all': 'Todos los satélites disponibles'
    }
    
    # logger.info(f"Satélites seleccionados: {', '.join(satellites_list)}")
    # for sat in satellites_list:
    #     logger.info(f"  • {sat}: {satellite_info.get(sat, 'Satélite no reconocido')}")
    

    
    # Mostrar información sobre sistema multi-escena
    # if args.use_multi_scene:
    #     logger.info("Sistema multi-escena habilitado: cada polígono se procesa con la mejor escena que lo cubra")
    # else:
    #     logger.info("Sistema tradicional: 1 escena PRE + 1 escena POST por tile")
    
    # Mostrar información sobre los parámetros configurados
    # logger.info(f"Tamaño de tile: {args.tile_size}x{args.tile_size} píxeles")
    # logger.info(f"Buffer alrededor de edificios: {args.buffer_meters} metros")
    
    # Mostrar información sobre las mejoras de calidad implementadas
    # logger.info("✅ Mejoras de calidad implementadas:")
    # logger.info("  • Remuestreo único directo con rasterio (sin doble interpolación)")
    # logger.info("  • Resampling adaptativo: AREA para downsample, BILINEAR para upsample")
    # logger.info("  • TARGET_GSD optimizado para preservar máxima resolución")
    # logger.info("  • Eliminación de cv2.resize innecesario cuando scale == 1.0")
    # logger.info("  • Normalización por percentiles por sensor y banda")
    # logger.info("  • Mejora automática de calidad visual (contraste y brillo)")
    
    # Mostrar información sobre las optimizaciones de rendimiento
    # logger.info("🚀 Optimizaciones de rendimiento implementadas:")
    # logger.info("  • Cache de datasets de rasterio para evitar aperturas repetidas")
    # logger.info("  • Rasterización de máscara una sola vez por tile")
    # logger.info("  • Variables globales pasadas a workers via initializer")
    # logger.info("  • Threads de BLAS/OpenCV limitados para evitar conflictos")
    # logger.info("  • Procesamiento con as_completed para mejor paralelismo")
    # logger.info("  • Sistema multi-escena para maximizar cobertura de polígonos")
    

    
    # Mostrar información sobre el procesamiento de bandas
    # logger.info("📊 Procesamiento de bandas:")
    # logger.info("  • Priorización de bandas _aligned cuando están disponibles")
    # logger.info("  • Fallback inteligente a bandas sustitutas (coastal → green → red)")
    # logger.info("  • Reordenamiento automático a RGB estándar")
    # logger.info("  • Soporte para múltiples sensores: WorldView, Pleiades, Kompsat, etc.")
    
    output_root = Path(args.output_dir)
    if output_root.exists() and output_root.is_dir():
        shutil.rmtree(output_root)
    data_root = Path(args.data_root)
    training_annotations = data_root / "Training" / "Annotations"
    training_data = data_root / "Training"
    output_root.mkdir(exist_ok=True)
    # === CALCULAR PERCENTILES GLOBALES ===
    # logger.info("Calculando percentiles globales por banda y sensor...")
    P = sample_patches_for_percentiles_per_sensor(training_data, sample_size=PERCENTILE_SAMPLE_SIZE)
    # logger.info(f"Percentiles calculados para {len(P)} combinaciones sensor-banda")
    gpkg_files = []
    for root, _, files in os.walk(training_annotations):
        for file in files:
            if file.lower().endswith(('training.gpkg', 'testing.gpkg')):
                gpkg_files.append(Path(root) / file)
    # logger.info(f"Encontrados {len(gpkg_files)} archivos GPKG para procesar")
    
    # Filtrar por país si se especifica
    if args.country:
        country_filter = args.country.strip().lower()
        valid_countries = set([v.lower() for v in EVENT_TO_COUNTRY.values()])
        if country_filter not in valid_countries:
            logger.error(f"País '{args.country}' no reconocido. Opciones válidas: {sorted(EVENT_TO_COUNTRY.values())}")
            return
        gpkg_files = [gpkg for gpkg in gpkg_files 
                     if EVENT_TO_COUNTRY.get(gpkg.parent.name, gpkg.parent.name).lower() == country_filter]
        if not gpkg_files:
            logger.warning(f"No se encontraron archivos GPKG para el país '{args.country}'")
            return
        # logger.info(f"Filtrados {len(gpkg_files)} archivos GPKG para el país '{args.country}'")
    
    # for gpkg in gpkg_files:
    #     logger.info(f"  - {gpkg.name}")
    if args.limit:
        gpkg_files = gpkg_files[:args.limit]

    # gpkg_files=gpkg_files[1:3]

    all_meta = []
    total_processed = 0
    total_failed = 0
    
    # logger.info(f"Procesando {len(gpkg_files)} archivos GPKG...")
    
    for i, gpkg_path in enumerate(gpkg_files):
        fname = gpkg_path.name.lower()
        if fname.endswith("training.gpkg"):
            split = 'full'
        elif fname.endswith("testing.gpkg"):
            split = 'test'
        else:
            split = 'unknown'
        
        # logger.info(f"[{i+1}/{len(gpkg_files)}] Procesando: {gpkg_path.name}")
        meta = process_gpkg(gpkg_path, training_data, output_root, split, args.buffer_meters, args.max_workers, args.satellites, args.tile_size, args.use_multi_scene, args.target_gsd)
        
        if meta:
            all_meta.extend(meta)
            total_processed += len(meta)
            logger.info(f"✅ {gpkg_path.name}: {len(meta)} tiles procesados exitosamente")
        else:
            total_failed += 1
            logger.warning(f"❌ {gpkg_path.name}: No se pudieron procesar tiles")
    
    logger.info(f"Resumen final: {total_processed} tiles procesados, {total_failed} archivos fallidos")
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