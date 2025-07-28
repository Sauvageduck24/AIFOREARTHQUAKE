import argparse
import logging
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
from pathlib import Path
import json
import re
from datetime import datetime
from shapely.geometry import shape, box
import numpy as np
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

def setup_logging():
    Path("outputs").mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler("outputs/salida_consola.txt", mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.info("=== INICIO DE NUEVA EJECUCIÓN ===")

setup_logging()

EVENT_TO_COUNTRY = {
    "Call_919_Turkey": "Turkey", "Call_1095_Myanmar": "Myanmar", "Call_1075_China": "China",
    "Call_920_Syria": "Syria", "Call_965_Morocco": "Morocco", "Call_970_Afghanistan": "Afghanistan"
}

COUNTRY_TO_CRS = {
    "Turkey": "EPSG:32637", "Myanmar": "EPSG:32646", "China": "EPSG:32645",
    "Syria": "EPSG:32637", "Morocco": "EPSG:32629", "Afghanistan": "EPSG:32642"
}

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
        
        logging.info(f"Leyendo JSON desde: {json_path}")
        with open(json_path, 'r') as f:
            item_data = json.load(f)
        
        assets = item_data.get('assets', {})
        logging.info(f"Assets encontrados en JSON: {list(assets.keys())}")
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
                            logging.info(f"Archivo encontrado en ubicación alternativa: {possible_path}")
                            break
                
                # Buscar archivo _aligned.tif primero
                aligned_asset_path = base_asset_path.parent / f"{base_asset_path.stem}_aligned.tif"
                
                if aligned_asset_path.exists():
                    available_bands[color] = aligned_asset_path
                    logging.info(f"Archivo alineado disponible para banda {color}: {aligned_asset_path.name}")
                elif base_asset_path.exists():
                    available_bands[color] = base_asset_path
                    logging.info(f"Archivo original disponible para banda {color}: {base_asset_path.name}")
                else:
                    logging.warning(f"No se encontró archivo para banda {color}: {base_asset_path}")
            else:
                logging.warning(f"No se encontró asset para banda {color}")
        
        # Si no tenemos al menos una banda, fallar
        if not available_bands:
            logging.error("No se encontraron archivos para ninguna banda RGB")
            return None, None
        
        bands = []
        transform = None
        
        # Intentar leer las bandas disponibles
        for color in ['red', 'green', 'blue']:
            if color in available_bands:
                asset_path = available_bands[color]
                logging.info(f"Leyendo banda {color} desde: {asset_path}")
                with rasterio.open(asset_path) as src:
                    if transform is None:
                        transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
                    band_data = src.read(1, out_shape=out_shape)
                    logging.info(f"Banda {color} leída exitosamente, forma: {band_data.shape}, rango: [{band_data.min():.2f}, {band_data.max():.2f}]")
                    bands.append(band_data)
            else:
                # Si falta una banda, usar la primera disponible como sustituto
                first_available = list(available_bands.keys())[0]
                asset_path = available_bands[first_available]
                logging.info(f"Usando banda {first_available} como sustituto para {color} desde: {asset_path}")
                with rasterio.open(asset_path) as src:
                    if transform is None:
                        transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
                    band_data = src.read(1, out_shape=out_shape)
                    logging.info(f"Banda sustituta {first_available} leída exitosamente, forma: {band_data.shape}")
                    bands.append(band_data)
        
        return np.stack(bands, axis=-1), transform
        
    except Exception as e:
        logging.error(f"Error leyendo bandas RGB: {e}")
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
                    logging.warning(f"No se pudo encontrar banda {color} en {tiff_path}")
                    return None, None
            
            img_array = np.stack(bands, axis=-1)
            transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
            return img_array, transform
            
    except Exception as e:
        logging.error(f"Error leyendo TIFF: {e}")
        return None, None

def _normalize_for_display(img_array):
    """Normaliza un array para visualización."""
    logging.info(f"Normalizando imagen con forma: {img_array.shape}, tipo: {img_array.dtype}")
    logging.info(f"Rango original: [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
        logging.info(f"Expandida a 3D, nueva forma: {img_array.shape}")
    
    # Manejar imágenes con 2 canales (crear un tercer canal)
    if img_array.shape[2] == 2:
        # Crear un tercer canal como promedio de los dos existentes
        third_channel = np.mean(img_array, axis=2, keepdims=True)
        img_array = np.concatenate([img_array, third_channel], axis=2)
        logging.info(f"Convertida de 2 a 3 canales, nueva forma: {img_array.shape}")
    elif img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
        logging.info(f"Repetida a 3 canales, nueva forma: {img_array.shape}")
    elif img_array.shape[2] > 3:
        # Si hay más de 3 canales, tomar solo los primeros 3
        img_array = img_array[:, :, :3]
        logging.info(f"Reducida a 3 canales, nueva forma: {img_array.shape}")

    img_norm = np.zeros_like(img_array, dtype=np.float32)
    for i in range(img_array.shape[2]):
        band = img_array[:, :, i]
        if np.any(band > 0):
            p2, p98 = np.percentile(band[band > 0], (2, 98))
            logging.info(f"Banda {i}: percentiles 2-98: [{p2:.2f}, {p98:.2f}]")
            if p98 > p2:
                img_norm[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                img_norm[:, :, i] = band.astype(np.float32) / (np.max(band) if np.max(band)>0 else 1)
        else:
            img_norm[:, :, i] = 0
            logging.info(f"Banda {i}: todos los valores son 0")
    
    logging.info(f"Imagen normalizada, rango final: [{img_norm.min():.2f}, {img_norm.max():.2f}]")
    return img_norm

def extract_earthquake_dates_from_collection(collection_path):
    """Extrae las fechas de terremotos desde el collection.json y carpetas de archivos."""
    earthquake_dates = {}
    
    try:
        with open(collection_path, 'r', encoding='utf-8') as f:
            collection_data = json.load(f)
        
        # Buscar fechas en los nombres de carpetas y archivos
        training_root = collection_path.parent
        for folder in training_root.iterdir():
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
                        logging.info(f"Fecha de terremoto detectada para {country}: {earthquake_date.strftime('%Y-%m-%d')}")
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
                                    logging.info(f"Fecha de terremoto detectada para {country} desde {json_file.name}: {earthquake_date.strftime('%Y-%m-%d')}")
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
                                    logging.info(f"Fecha de terremoto detectada para {country} desde propiedades: {earthquake_date.strftime('%Y-%m-%d')}")
                                    break
                                except ValueError:
                                    pass
                            
                            # Buscar fecha en el título de las propiedades
                            title = properties.get('title', '')
                            if title:
                                # Buscar patrones como "2023-06-26" o "26/06/2023"
                                date_patterns = [
                                    r'(\d{4}-\d{2}-\d{2})',  # 2023-06-26
                                    r'(\d{2}/\d{2}/\d{4})',  # 26/06/2023
                                    r'(\d{8})',              # 20230626
                                ]
                                
                                for pattern in date_patterns:
                                    date_match = re.search(pattern, title)
                                    if date_match:
                                        date_str = date_match.group(1)
                                        try:
                                            if '-' in date_str:
                                                earthquake_date = datetime.strptime(date_str, '%Y-%m-%d')
                                            elif '/' in date_str:
                                                earthquake_date = datetime.strptime(date_str, '%d/%m/%Y')
                                            else:
                                                earthquake_date = datetime.strptime(date_str, '%Y%m%d')
                                            
                                            earthquake_dates[country] = earthquake_date
                                            logging.info(f"Fecha de terremoto detectada para {country} desde título: {earthquake_date.strftime('%Y-%m-%d')}")
                                            break
                                        except ValueError:
                                            continue
                                    
                        except Exception as e:
                            continue
                            
    except Exception as e:
        logging.error(f"Error extrayendo fechas de terremotos: {e}")
    
    # Fallback a fechas conocidas si no se pudieron extraer
    if not earthquake_dates:
        earthquake_dates = {
            "Turkey": datetime(2023, 2, 6), "Syria": datetime(2023, 2, 6), 
            "Morocco": datetime(2023, 9, 8), "Afghanistan": datetime(2023, 10, 7),
            "China": datetime(2025, 1, 7), "Myanmar": datetime(2025, 3, 27)
        }
        logging.info("Usando fechas de terremotos por defecto")
    else:
        logging.info(f"Fechas de terremotos extraídas: {list(earthquake_dates.keys())}")
    
    return earthquake_dates

def find_scenes_for_aoi(aoi_geom, training_root, satellites=None):
    """Encuentra escenas usando collection.json y metadatos de TIFF."""
    # Para encontrar escenas pre/post terremoto, no filtramos por AOI específico
    # ya que necesitamos todas las escenas disponibles para el país
    training_root = Path(training_root)
    collection_path = training_root / "collection.json"
    
    # Procesar lista de satélites
    if satellites is None:
        satellites = ['wv', 'ds']
    elif isinstance(satellites, str):
        satellites = [s.strip().lower() for s in satellites.split(',')]
    
    logging.info(f"Filtrando escenas por satélites: {satellites}")

    if not collection_path.exists():
        logging.warning(f"No se encontró collection.json en {training_root}")
        return []

    # Extraer fechas de terremotos
    earthquake_dates = extract_earthquake_dates_from_collection(collection_path)

    with open(collection_path, 'r', encoding='utf-8') as f:
        collection_data = json.load(f)

    all_item_data = []
    item_links = [link['href'] for link in collection_data.get('links', []) if link.get('rel') == 'item']
    logging.info(f"Encontrados {len(item_links)} items en collection.json")
    for item_href in item_links:
        item_path = (collection_path.parent / item_href).resolve()
        logging.info(f"Procesando item: {item_href}")
        if item_path.exists():
            try:
                with open(item_path, 'r', encoding='utf-8') as f:
                    item_data = json.load(f)
                    item_data['__item_path'] = item_path
                    all_item_data.append(item_data)
                    logging.info(f"Item cargado exitosamente: {item_data.get('id', 'sin_id')}")
            except Exception as e:
                logging.error(f"Error leyendo item JSON {item_path}: {e}")
        else:
            logging.warning(f"Item no encontrado: {item_path}")

    grouped_scenes = {}
    logging.info(f"Agrupando {len(all_item_data)} items...")
    for item in all_item_data:
        scene_id = item.get('id', '')
        base_name = scene_id
        if 'WV03N28_549166E087' in scene_id:
            base_name = 'WV03N28_549166E087'
        else:
            if re.search(r"MS\d{2}", scene_id):
                base_name = scene_id.split('_MS')[0]
            elif re.search(r"PN\d{2}", scene_id):
                base_name = scene_id.split('_PN')[0]
            if base_name.endswith('-calibrated'):
                base_name = base_name[:-11]
        
        logging.info(f"Scene ID: {scene_id} -> Base name: {base_name}")
        
        if base_name not in grouped_scenes:
            grouped_scenes[base_name] = {}
        
        if re.search(r"MS\d{2}", scene_id):
            grouped_scenes[base_name]['MS'] = item
            logging.info(f"Agregado como MS: {scene_id}")
        elif re.search(r"PN\d{2}", scene_id):
            grouped_scenes[base_name]['PN'] = item
            logging.info(f"Agregado como PN: {scene_id}")
        else:
            grouped_scenes[base_name]['OTRO'] = item
            logging.info(f"Agregado como OTRO: {scene_id}")

    final_items_to_process = []
    logging.info(f"Procesando {len(grouped_scenes)} grupos de escenas...")
    for base_name, items in grouped_scenes.items():
        is_worldview = 'wv' in base_name.lower() or 'worldview' in base_name.lower()
        logging.info(f"Grupo {base_name}: WorldView={is_worldview}, items disponibles: {list(items.keys())}")
        if is_worldview:
            if 'MS' in items:
                final_items_to_process.append(items['MS'])
                logging.info(f"Seleccionada escena MS para WorldView: {items['MS'].get('id')}")
            else:
                logging.warning(f"No se encontró escena MS para WorldView {base_name}")
        else:
            final_items_to_process.extend(items.values())
            logging.info(f"Agregadas todas las escenas para {base_name}: {[item.get('id') for item in items.values()]}")
    
    logging.info(f"Total de escenas finales a procesar: {len(final_items_to_process)}")

    scenes = []
    logging.info(f"Procesando {len(final_items_to_process)} escenas finales...")
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
                        logging.info(f"Escena {scene_id} coincide con satélite '{satellite}' (patrón: {pattern})")
                        break
                if satellite_match:
                    break
            
            if not satellite_match:
                logging.info(f"Escena {scene_id} descartada - no coincide con satélites solicitados: {satellites}")
                continue

            logging.info(f"Procesando escena: {scene_id} (Fecha: {scene_date}, Plataforma: {platform})")

            overview_asset = None
            # Buscar cualquier asset que contenga "overview" en su clave
            for key, asset in assets.items():
                if 'overview' in key.lower() and 'href' in asset:
                    overview_asset = asset
                    logging.info(f"Encontrado asset de overview: {key}")
                    break
            
            # Si no se encuentra, buscar assets con rol 'visual'
            if not overview_asset:
                for key, asset in assets.items():
                    if 'visual' in asset.get('roles', []) and 'href' in asset:
                        overview_asset = asset
                        logging.info(f"Encontrado asset visual: {key}")
                        break
            
            if not overview_asset:
                logging.warning(f"No se encontró asset de overview para {scene_id}")
                continue

            # Construir la ruta correctamente
            # item_path.parent es la carpeta de la escena
            scene_folder = item_path.parent
            asset_href = overview_asset['href']
            
            # Extraer solo el nombre del archivo del href para evitar duplicación de rutas
            asset_filename = Path(asset_href).name
            asset_stem = Path(asset_href).stem
            asset_suffix = Path(asset_href).suffix
            
            # Crear nombres de archivo con y sin "_aligned"
            aligned_filename = f"{asset_stem}_aligned{asset_suffix}"
            
            # Intentar diferentes ubicaciones posibles para el archivo, priorizando _aligned
            possible_base_paths = [
                scene_folder / asset_href,  # Ruta original como viene en el href
                scene_folder / asset_filename,  # Solo nombre del archivo en la carpeta de la escena
                scene_folder / "Optical_Calibration" / asset_filename,  # En subcarpeta Optical_Calibration
            ]
            
            overview_path = None
            
            # Primero buscar versiones _aligned
            for base_path in possible_base_paths:
                aligned_path = base_path.parent / aligned_filename
                if aligned_path.exists():
                    overview_path = aligned_path.resolve()
                    logging.info(f"Archivo de overview alineado encontrado en: {overview_path}")
                    break
            
            # Si no se encuentra _aligned, buscar archivos originales
            if overview_path is None:
                for possible_path in possible_base_paths:
                    if possible_path.exists():
                        overview_path = possible_path.resolve()
                        logging.info(f"Archivo de overview original encontrado en: {overview_path}")
                        break
            
            if overview_path is None:
                logging.warning(f"No se encontró archivo de overview con href: {asset_href}")
                # Buscar archivos TIFF en la carpeta de la escena
                tiff_files = list(scene_folder.glob("*.tif")) + list(scene_folder.glob("*.tiff"))
                if tiff_files:
                    overview_path = tiff_files[0]
                    logging.info(f"Usando archivo TIFF encontrado: {overview_path.name}")
                else:
                    # Buscar en subcarpetas
                    for subfolder in scene_folder.iterdir():
                        if subfolder.is_dir():
                            tiff_files = list(subfolder.glob("*.tif")) + list(subfolder.glob("*.tiff"))
                            if tiff_files:
                                overview_path = tiff_files[0]
                                logging.info(f"Usando archivo TIFF encontrado en subcarpeta: {overview_path.name}")
                                break
                    else:
                        logging.warning(f"No se encontraron archivos TIFF para {scene_id}")
                        continue
            
            # Determinar el path de la escena correctamente
            if "Optical_Calibration" in str(overview_path):
                scene_path = overview_path.parent.parent  # Salir de Optical_Calibration y de la carpeta de la escena
            else:
                scene_path = overview_path.parent
            
            # Extraer información geoespacial directamente del archivo TIFF
            scene_geom = None
            scene_crs = None
            scene_gsd = None
            scene_shape = None
            
            # Intentar obtener información del overview_path
            try:
                with rasterio.open(overview_path) as src:
                    # Obtener CRS del TIFF
                    scene_crs = str(src.crs) if src.crs else None
                    
                    # Obtener forma del TIFF
                    scene_shape = [src.height, src.width]
                    
                    # Calcular GSD aproximado desde el transform
                    if src.transform:
                        scene_gsd = abs(src.transform[0])  # Resolución en X
                    
                    # Crear geometría desde bounds del TIFF
                    bounds = src.bounds
                    scene_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                    
                    # Si el CRS no es WGS84, reprojectar la geometría
                    if src.crs and src.crs != 'EPSG:4326':
                        try:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                            scene_geom = shapely_transform(transformer.transform, scene_geom)
                        except Exception as e:
                            logging.warning(f"No se pudo reprojectar geometría: {e}")
                    
                    logging.info(f"Información extraída del TIFF - CRS: {scene_crs}, Shape: {scene_shape}, GSD: {scene_gsd:.2f}" if scene_gsd else f"Información extraída del TIFF - CRS: {scene_crs}, Shape: {scene_shape}")
                    
            except Exception as e:
                logging.warning(f"No se pudo extraer información del TIFF {overview_path}: {e}")
                # Fallback a metadatos del JSON si falla la lectura del TIFF
                scene_geom = shape(item_data['geometry']) if 'geometry' in item_data else None
                scene_crs = f"EPSG:{properties['proj:epsg']}" if properties.get('proj:epsg') is not None else None
                scene_gsd = properties.get('gsd')
                scene_shape = properties.get('proj:shape')
            
            scene = {
                'scene_id': scene_id,
                'date': scene_date,
                'geometry': scene_geom,
                'overview_path': overview_path,
                'path': scene_path,
                'crs': scene_crs,
                'gsd': scene_gsd,
                'shape': scene_shape,
                'earthquake_dates': earthquake_dates  # Pasar las fechas a cada escena
            }
            scenes.append(scene)
            logging.info(f"Escena añadida: {scene_id} (Plataforma: {properties.get('platform', 'N/A')})")
        
        except Exception as e:
            logging.error(f"Error procesando escena {item_data.get('id', 'desconocida')}: {e}")
            continue
    
    logging.info(f"Total de escenas procesadas: {len(scenes)}")

    return scenes

def parse_date_from_scene_id(scene_id: str):
    """Extrae fecha de un identificador de escena."""
    m14 = re.search(r"(20\d{12})", scene_id)
    if m14:
        try:
            return datetime.strptime(m14.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    m8 = re.search(r"(20\d{6})", scene_id)
    if m8:
        try:
            return datetime.strptime(m8.group(1), "%Y%m%d")
        except ValueError:
            pass
    return None

def find_all_gpkg_files(data_root):
    """Encuentra todos los archivos GPKG."""
    training_annotations = Path(data_root) / "Training" / "Annotations"
    gpkg_files = []
    for root, _, files in os.walk(training_annotations):
        for file in files:
            if file.lower().endswith(('training.gpkg', 'testing.gpkg')):
                gpkg_files.append(Path(root) / file)
    return gpkg_files



def clean_and_validate_geometries(gdf):
    """Limpia y valida geometrías."""
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    valid_geometries = []
    for geom in gdf.geometry:
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_valid and not geom.is_empty:
                valid_geometries.append(geom)
        except Exception as e:
            continue
    
    if valid_geometries:
        return gpd.GeoDataFrame(geometry=valid_geometries, crs=gdf.crs)
    return None

def get_aoi_geometry(gdf):
    """Obtiene geometría del área de interés."""
    try:
        return gdf.to_crs(4326).geometry.union_all()
    except Exception as e:
        gdf_clean = clean_and_validate_geometries(gdf)
        if gdf_clean is None:
            return None
        try:
            return gdf_clean.to_crs(4326).geometry.union_all()
        except Exception as e2:
            try:
                bbox = gdf_clean.to_crs(4326).total_bounds
                return box(bbox[0], bbox[1], bbox[2], bbox[3])
            except Exception as e3:
                return None

def extract_city_from_gpkg(gpkg_path):
    """Extrae la ciudad del nombre del archivo GPKG."""
    name = gpkg_path.name.replace('.gpkg', '')
    city = re.split(r'[_-]', name)[0]
    return city.capitalize()

def calculate_buffer_meters(scene_id, gsd=None, user_buffer=None):
    """Calcula el buffer en metros según el tipo de sensor."""
    if user_buffer is not None:
        return user_buffer
    
    scene_id_upper = scene_id.upper()
    if any(sensor in scene_id_upper for sensor in ['DS_PHR', 'PLEIADES']):
        return 20.0
    elif any(sensor in scene_id_upper for sensor in ['WV', 'WORLDVIEW', 'GE', 'GEOEYE']):
        return 40.0
    else:
        return 60.0

def visualize_scene_with_polygons(scene, gpkg_path, layer_name, country, city=None, show_polygons=True, filter_outside_polygons=True):
    """Visualiza una escena con polígonos superpuestos."""
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        gdf_clean = clean_and_validate_geometries(gdf)
        if gdf_clean is None:
            return
        
        with rasterio.open(scene['overview_path']) as src:
            img_crs = src.crs
            img_bounds = src.bounds
            img_transform = src.transform
            img_width = src.width
            img_height = src.height
            
            logging.info(f"Información del TIFF - CRS: {img_crs}, Bounds: {img_bounds}, Transform: {img_transform}")
            
            if gdf_clean.crs != img_crs:
                try:
                    gdf_clean = gdf_clean.to_crs(img_crs)
                    logging.info(f"GPKG reproyectado de {gdf.crs} a {img_crs}")
                except Exception as e:
                    logging.error(f"Error reproyectando GPKG: {e}")
                    return

            has_rotation = not (abs(src.transform.b) < 1e-9 and abs(src.transform.d) < 1e-9)

            if has_rotation:
                img = src.read(1, out_shape=(src.height // 4, src.width // 4))
                img_transform = src.transform * src.transform.scale(src.width / (src.width // 4), src.height / (src.height // 4))
                img = _normalize_for_display(img)
            else:
                scale_factor = 4
                out_shape = (src.height // scale_factor, src.width // scale_factor)
                
                img, img_transform = _read_rgb_bands_from_separate_files(scene['path'], scene['scene_id'], out_shape)
                
                if img is None:
                    img, img_transform = _read_rgb_bands_from_tiff(scene['overview_path'], out_shape)

                if img is None:
                    num_bands = min(3, src.count)
                    data = src.read(list(range(1, num_bands + 1)), out_shape=out_shape)
                    img = np.moveaxis(data, 0, -1)
                    img_transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])

                img = _normalize_for_display(img)
            
            # Usar los bounds ya extraídos del TIFF
            left, bottom, right, top = img_bounds.left, img_bounds.bottom, img_bounds.right, img_bounds.top
            logging.info(f"Bounds de visualización - Left: {left}, Bottom: {bottom}, Right: {right}, Top: {top}")
            
            logging.info(f"Creando figura matplotlib con imagen de forma: {img.shape}")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img, extent=(left, right, bottom, top))
            logging.info("Imagen mostrada en matplotlib")

            patches = []
            shown = 0
            centroids = []
            
            max_polys = 100
            if show_polygons and len(gdf_clean) > max_polys:
                gdf_clean = gdf_clean.sample(max_polys, random_state=42)
            
            if show_polygons and not gdf_clean.empty:
                auto_buffer = calculate_buffer_meters(scene['scene_id'], scene.get('gsd'))
                original_crs = gdf_clean.crs
                if gdf_clean.crs.is_geographic:
                    centroid = gdf_clean.geometry.union_all().centroid
                    lon, lat = centroid.x, centroid.y
                    utm_zone = int((lon + 180) / 6) + 1
                    utm_crs = f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"
                    gdf_clean = gdf_clean.to_crs(utm_crs)
                    gdf_clean.geometry = gdf_clean.geometry.buffer(auto_buffer)
                    gdf_clean = gdf_clean.to_crs(original_crs)
                else:
                    gdf_clean.geometry = gdf_clean.geometry.buffer(auto_buffer)
            
            if show_polygons and not gdf_clean.empty and filter_outside_polygons:
                from shapely.geometry import box
                img_bbox = box(left, bottom, right, top)
                gdf_clean = gdf_clean[gdf_clean.geometry.intersects(img_bbox)]
                
                if gdf_clean.empty:
                    show_polygons = False
            
            if show_polygons:
                for geom in gdf_clean.geometry:
                    if shown >= max_polys:
                        break
                    try:
                        if geom.geom_type == 'Polygon':
                            patches.append(Polygon(list(geom.exterior.coords), closed=True, fill=False, edgecolor='red', linewidth=1.5))
                            centroids.append(geom.centroid)
                            shown += 1
                        elif geom.geom_type == 'MultiPolygon':
                            for poly in geom.geoms:
                                if shown >= max_polys:
                                    break
                                patches.append(Polygon(list(poly.exterior.coords), closed=True, fill=False, edgecolor='red', linewidth=1.5))
                                centroids.append(poly.centroid)
                                shown += 1
                    except Exception as e:
                        continue
                
                if patches:
                    ax.add_collection(PatchCollection(patches, match_original=True))
                    for idx, c in enumerate(centroids):
                        ax.plot(c.x, c.y, 'o', color='red', markersize=8)
                        ax.text(c.x, c.y, str(idx+1), color='yellow', fontsize=8, fontweight='bold', ha='center', va='center')
            else:
                ax.text(0.5, 0.95, 'ESCENA PRE-TERREMOTO\nLos polígonos de daños no deberían aparecer aquí', 
                       transform=ax.transAxes, fontsize=14, ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            date_str = scene['date'].strftime('%Y-%m-%d %H:%M') if scene['date'] else 'Fecha desconocida'
            img_epsg = src.crs.to_epsg() if src.crs else 'No definido'
            poly_epsg = gdf.crs.to_epsg() if gdf.crs else 'No definido'
            polygon_status = "MOSTRADOS" if show_polygons else "OCULTOS (PRE-TERREMOTO)"
            
            logging.info(f"Visualizando imagen con CRS del TIFF: {img_epsg}, Polígonos con CRS: {poly_epsg}")
            
            pre_post_status = "DESCONOCIDO"
            pre_post_color = "gray"
            if scene['date'] and 'earthquake_dates' in scene:
                earthquake_dates = scene['earthquake_dates']
                
                if country in earthquake_dates:
                    earthquake_date = earthquake_dates[country]
                    scene_date_naive = scene['date'].replace(tzinfo=None) if scene['date'].tzinfo else scene['date']
                    earthquake_date_naive = earthquake_date.replace(tzinfo=None) if earthquake_date.tzinfo else earthquake_date
                    
                    if scene_date_naive < earthquake_date_naive:
                        pre_post_status = "PRE-TERREMOTO"
                        pre_post_color = "orange"
                    else:
                        pre_post_status = "POST-TERREMOTO"
                        pre_post_color = "red"
            
            city_line = f"Ciudad: {city}" if city else ""
            title = f"País: {country}\n{city_line}\nEscena: {scene['scene_id']}\nFecha: {date_str}\nCRS Imagen (TIFF): EPSG:{img_epsg} | CRS Polígonos: EPSG:{poly_epsg}\nPolígonos: {polygon_status}"
            plt.title(title, fontsize=10)
            
            ax.text(0.02, 0.98, pre_post_status, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=pre_post_color, alpha=0.8, edgecolor='black'),
                   color='white')
            
            plt.tight_layout()
            logging.info("Mostrando figura matplotlib...")
            plt.show()
            logging.info("Figura matplotlib cerrada")
            
    except Exception as e:
        logging.error(f"Error visualizando escena {scene['scene_id']}: {e}")

def visualize_pre_post_scenes_with_polygons(pre_scene, post_scene, gpkg_path, layer_name, country, city=None, show_polygons=True, filter_outside_polygons=True):
    """Visualiza lado a lado la escena pre y post-terremoto con los polígonos superpuestos."""
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    gdf_clean = clean_and_validate_geometries(gdf)
    if gdf_clean is None:
        return

    scenes = [pre_scene, post_scene]
    titles = ["PRE-TERREMOTO", "POST-TERREMOTO"]
    imgs = []
    img_extents = []
    img_crss = []
    gdfs = []
    scale_factor = 4
    
    for idx, scene in enumerate(scenes):
        with rasterio.open(scene['overview_path']) as src:
            img_crs = src.crs
            img_bounds = src.bounds
            img_transform = src.transform
            
            img_crss.append(img_crs)
            gdf_proj = gdf_clean
            if gdf_clean.crs != img_crs:
                try:
                    gdf_proj = gdf_clean.to_crs(img_crs)
                    logging.info(f"GPKG reproyectado de {gdf.crs} a {img_crs} para escena {idx}")
                except Exception as e:
                    logging.error(f"Error reproyectando GPKG: {e}")
                    return
                    
            has_rotation = not (abs(src.transform.b) < 1e-9 and abs(src.transform.d) < 1e-9)
            if has_rotation:
                img = src.read(1, out_shape=(src.height // scale_factor, src.width // scale_factor))
                img = _normalize_for_display(img)
                img_bounds = src.bounds
            else:
                out_shape = (src.height // scale_factor, src.width // scale_factor)
                
                img, _ = _read_rgb_bands_from_separate_files(scene['path'], scene['scene_id'], out_shape)
                
                if img is None:
                    img, _ = _read_rgb_bands_from_tiff(scene['overview_path'], out_shape)
                
                if img is None:
                    num_bands = min(3, src.count)
                    data = src.read(list(range(1, num_bands + 1)), out_shape=out_shape)
                    img = np.moveaxis(data, 0, -1)
                
                img = _normalize_for_display(img)
                
            imgs.append(img)
            # Usar los bounds ya extraídos del TIFF
            img_extents.append((img_bounds.left, img_bounds.right, img_bounds.bottom, img_bounds.top))
            gdfs.append(gdf_proj)
            
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for idx, ax in enumerate(axes):
        img = imgs[idx]
        left, right, bottom, top = img_extents[idx]
        ax.imshow(img, extent=(left, right, bottom, top))
        gdf_proj = gdfs[idx]
        
        patches = []
        centroids = []
        max_polys = 100
        gdf_show = gdf_proj
        if show_polygons and len(gdf_proj) > max_polys:
            gdf_show = gdf_proj.sample(max_polys, random_state=42)
            
        if show_polygons and not gdf_show.empty:
            auto_buffer = calculate_buffer_meters(scenes[idx]['scene_id'], scenes[idx].get('gsd'))
            original_crs = gdf_show.crs
            if gdf_show.crs.is_geographic:
                centroid = gdf_show.geometry.union_all().centroid
                lon, lat = centroid.x, centroid.y
                utm_zone = int((lon + 180) / 6) + 1
                utm_crs = f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"
                gdf_show = gdf_show.to_crs(utm_crs)
                gdf_show.geometry = gdf_show.geometry.buffer(auto_buffer)
                gdf_show = gdf_show.to_crs(original_crs)
            else:
                gdf_show.geometry = gdf_show.geometry.buffer(auto_buffer)
                
        if show_polygons and not gdf_show.empty and filter_outside_polygons:
            from shapely.geometry import box
            img_bbox = box(left, bottom, right, top)
            gdf_show = gdf_show[gdf_show.geometry.intersects(img_bbox)]
            
        shown = 0
        for geom in gdf_show.geometry:
            if shown >= max_polys:
                break
            try:
                if geom.geom_type == 'Polygon':
                    patches.append(Polygon(list(geom.exterior.coords), closed=True, fill=False, edgecolor='red', linewidth=1.5))
                    centroids.append(geom.centroid)
                    shown += 1
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        if shown >= max_polys:
                            break
                        patches.append(Polygon(list(poly.exterior.coords), closed=True, fill=False, edgecolor='red', linewidth=1.5))
                        centroids.append(poly.centroid)
                        shown += 1
            except Exception as e:
                continue
                
        if patches:
            ax.add_collection(PatchCollection(patches, match_original=True))
            for idx2, c in enumerate(centroids):
                ax.plot(c.x, c.y, 'o', color='red', markersize=8)
                
        date_str = scenes[idx]['date'].strftime('%Y-%m-%d %H:%M') if scenes[idx]['date'] else 'Fecha desconocida'
        img_epsg = img_crss[idx].to_epsg() if img_crss[idx] else 'No definido'
        poly_epsg = gdf.crs.to_epsg() if gdf.crs else 'No definido'
        city_line = f"Ciudad: {city}" if city else ""
        ax.set_title(f"{titles[idx]}\nPaís: {country}\n{city_line}\nEscena: {scenes[idx]['scene_id']}\nFecha: {date_str}\nCRS Imagen (TIFF): EPSG:{img_epsg} | CRS Polígonos: EPSG:{poly_epsg}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualizador de escenas TIFF con polígonos de anotación.")
    parser.add_argument('--data-root', type=str, default="data/international-charter-eo4ai-etq-challenge", help='Ruta raíz de los datos.')
    parser.add_argument('--limit', type=int, default=None, help='Limitar número de GPKGs a procesar.')
    parser.add_argument('--country', type=str, default=None, help='Filtrar y procesar solo escenas de este país.')
    parser.add_argument('--filter-outside-polygons', action='store_true', default=True, help='Filtrar polígonos que caen fuera de los límites de la imagen.')
    parser.add_argument('--all-scenes', action='store_true', help='Mostrar todas las escenas disponibles para el país, no solo las que intersectan con el GPKG.')
    parser.add_argument('--satellites', type=str, default='wv,ds', help='Satélites a usar (separados por coma). Opciones: wv,ds,pleiades,geoeye,ge. Por defecto: wv,ds')
    args = parser.parse_args()

    # Mostrar información sobre los satélites seleccionados
    satellites_list = [s.strip().lower() for s in args.satellites.split(',')]
    logging.info(f"Satélites seleccionados: {satellites_list}")
    
    gpkg_files = find_all_gpkg_files(args.data_root)
    logging.info(f"Encontrados {len(gpkg_files)} archivos GPKG")
    
    if args.limit:
        gpkg_files = gpkg_files[:args.limit]

    if args.country:
        country_filter = args.country.strip().lower()
        valid_countries = set([v.lower() for v in EVENT_TO_COUNTRY.values()])
        if country_filter not in valid_countries:
            logging.error(f"País '{args.country}' no reconocido. Opciones válidas: {sorted(EVENT_TO_COUNTRY.values())}")
            return
        gpkg_files = [gpkg for gpkg in gpkg_files 
                     if EVENT_TO_COUNTRY.get(gpkg.parent.name, gpkg.parent.name).lower() == country_filter]
        if not gpkg_files:
            logging.warning(f"No se encontraron archivos GPKG para el país '{args.country}'")
            return
    
    for gpkg_path in gpkg_files:
        logging.info(f"\nProcesando GPKG: {gpkg_path.name}")
        
        with fiona.Env():
            with fiona.open(gpkg_path) as gpkg:
                layer_name = gpkg.name
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        logging.info(f"Polígonos cargados: {len(gdf)}")
        
        aoi_geom = get_aoi_geometry(gdf)
        if aoi_geom is None:
            logging.warning(f"No se pudo obtener un área de interés válida para {gpkg_path.name}")
            continue
        
        event_name = gpkg_path.parent.name
        event_dir = Path(args.data_root) / "Training" / event_name
        country = EVENT_TO_COUNTRY.get(event_name, event_name)
        logging.info(f"País: {country}")
        
        if not event_dir.exists():
            logging.warning(f"Directorio del evento no encontrado: {event_dir}")
            continue
        
        scenes = find_scenes_for_aoi(aoi_geom, event_dir, args.satellites)
        if args.all_scenes:
            logging.info("Modo --all-scenes activado: mostrando todas las escenas disponibles para el país")
            # No filtrar por AOI, usar todas las escenas encontradas
        else:
            logging.info("Filtrando escenas por intersección con el área de interés del GPKG")
            # Filtrar escenas que intersectan con el AOI del GPKG
            if aoi_geom and scenes:
                filtered_scenes = []
                for scene in scenes:
                    if scene['geometry'] and scene['geometry'].intersects(aoi_geom):
                        filtered_scenes.append(scene)
                scenes = filtered_scenes
                logging.info(f"Escenas que intersectan con AOI: {len(scenes)}")
        
        logging.info(f"Escenas encontradas: {len(scenes)}")
        
        if not scenes:
            logging.warning(f"No se encontraron escenas para {event_name}")
            continue
        
        logging.info(f"Escenas encontradas: {len(scenes)}")
        
        if len(scenes) >= 2:
            scenes_sorted = sorted([s for s in scenes if s['date']], key=lambda s: s['date'])
            if len(scenes_sorted) >= 2:
                pre_scene, post_scene = scenes_sorted[0], scenes_sorted[-1]
                if pre_scene['scene_id'] != post_scene['scene_id']:
                    city = extract_city_from_gpkg(gpkg_path)
                    logging.info(f"Visualizando pre y post-terremoto lado a lado: {pre_scene['scene_id']} vs {post_scene['scene_id']}")
                    visualize_pre_post_scenes_with_polygons(pre_scene, post_scene, gpkg_path, layer_name, country, city, 
                                                          show_polygons=True, filter_outside_polygons=args.filter_outside_polygons)
                    continue
        
        for scene in scenes:
            logging.info(f"Visualizando escena: {scene['scene_id']}")
            city = extract_city_from_gpkg(gpkg_path)
            
            show_polygons = True
            if len(scenes) >= 2:
                dates = [s['date'] for s in scenes if s['date']]
                if dates:
                    scene_date_naive = scene['date'].replace(tzinfo=None) if scene['date'].tzinfo else scene['date']
                    if scene_date_naive == min(dates):
                        status = "PRE-TERREMOTO"
                    elif scene_date_naive == max(dates):
                        status = "POST-TERREMOTO"
                    else:
                        status = "INTERMEDIA"
                else:
                    status = "ÚNICA"
            else:
                status = "ÚNICA"
            
            logging.info(f"Visualizando escena {status}: {scene['scene_id']}")
            visualize_scene_with_polygons(scene, gpkg_path, layer_name, country, city, 
                                        show_polygons=show_polygons, filter_outside_polygons=args.filter_outside_polygons)
        
        logging.info(f"Completado GPKG: {gpkg_path.name}")

if __name__ == "__main__":
    main() 