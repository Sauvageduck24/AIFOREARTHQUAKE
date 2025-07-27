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
from datetime import datetime, timezone
from shapely.geometry import shape, box
import numpy as np
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
from shapely.ops import transform as shapely_transform
from pyproj import Transformer



def _read_rgb_bands(assets, base_path, out_shape):
    """
    Busca los assets de las bandas R, G, B, los lee y los apila en un array (H, W, 3).
    Usa 'common_name' o claves de asset para identificar las bandas.
    """
    rgb_assets = {}
    band_keys = {
        'red': ['red', 'B4', 'B04'],
        'green': ['green', 'B3', 'B03'],
        'blue': ['blue', 'B2', 'B02']
    }

    # Buscar assets por common_name o por clave
    for key, asset in assets.items():
        common_name = asset.get('eo:bands', [{}])[0].get('common_name')
        if common_name in band_keys:
            rgb_assets[common_name] = asset
        else:
            for color, keys in band_keys.items():
                if key in keys:
                    rgb_assets[color] = asset
                    break
    
    # Si no encontramos las 3 bandas, no podemos componer una imagen RGB
    if not all(k in rgb_assets for k in ['red', 'green', 'blue']):
        logging.warning("No se encontraron los assets para las 3 bandas RGB. Se volverá a la imagen de un solo canal.")
        return None, None
    
    # Leer cada banda
    bands = []
    for color in ['red', 'green', 'blue']:
        asset_href = rgb_assets[color]['href']
        asset_path = (base_path / asset_href).resolve()
        
        if not asset_path.exists():
            logging.error(f"El archivo de la banda {color} no existe: {asset_path}")
            return None, None
            
        with rasterio.open(asset_path) as src:
            # Guardar el transform de la primera banda que leamos
            if not bands:
                transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])
            
            bands.append(src.read(1, out_shape=out_shape))

    # Apilar bandas
    img_array = np.stack(bands, axis=-1)
    return img_array, transform


def _normalize_for_display(img_array):
    """Normaliza un array (alto, ancho, banda) para visualización con imshow."""
    # Si la imagen solo tiene una banda, expandirla para que funcione el resto del código
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Si es monocromática, la duplicamos en 3 canales para mostrarla en escala de grises
    if img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    img_norm = np.zeros_like(img_array, dtype=np.float32)
    for i in range(img_array.shape[2]):
        band = img_array[:, :, i]
        # Evitar error si todo el array es 0
        if np.any(band > 0):
            p2, p98 = np.percentile(band[band > 0], (2, 98))
            if p98 > p2:
                img_norm[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else: # Handle cases with no variation
                 img_norm[:, :, i] = band.astype(np.float32) / (np.max(band) if np.max(band)>0 else 1)
        else:
            img_norm[:, :, i] = 0 # All zeros
    return img_norm

# Configurar logging para consola y archivo
def setup_logging():
    # Crear directorio outputs si no existe
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Configurar el logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Formato de los mensajes
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (mode='w' para reescribir cada vez)
    file_handler = logging.FileHandler(output_dir / "salida_consola.txt", mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info("=== INICIO DE NUEVA EJECUCIÓN ===")
    logging.info(f"Salida de consola guardándose en: {output_dir / 'salida_consola.txt'}")

# Llamar a la configuración del logging
setup_logging()

EVENT_TO_COUNTRY = {
    "Call_919_Turkey": "Turkey",
    "Call_1095_Myanmar": "Myanmar", 
    "Call_1075_China": "China",
    "Call_920_Syria": "Syria",
    "Call_965_Morocco": "Morocco",
    "Call_970_Afghanistan": "Afghanistan"
}

# Mapeo de países a CRS por defecto
COUNTRY_TO_CRS = {
    "Turkey": "EPSG:32637",    # WGS 84 / UTM zone 37N
    "Myanmar": "EPSG:32646",   # WGS 84 / UTM zone 46N
    "China": "EPSG:32645",     # WGS 84 / UTM zone 45N
    "Syria": "EPSG:32637",     # WGS 84 / UTM zone 37N
    "Morocco": "EPSG:32629",   # WGS 84 / UTM zone 29N
    "Afghanistan": "EPSG:32642" # WGS 84 / UTM zone 42N
}

def get_country_from_event(event_name):
    return EVENT_TO_COUNTRY.get(event_name, event_name)

def find_scenes_for_aoi_legacy(aoi_geom, training_root):
    scenes = []
    training_root = Path(training_root)
    
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
            if fecha_base.startswith("WV") or "worldview" in fecha_base.lower():
                logging.info(f"Ignorando escena panchromática: {carpeta.name}")
                continue
            else:
                escenas_por_fecha[fecha_base]["PN"] = carpeta
        else:
            escenas_por_fecha[fecha_base]["OTRO"] = carpeta
    
    for fecha_base, paths in escenas_por_fecha.items():
        scene_geom = None
        scene_date = None
        overview_path = None
        scene_crs = None
        scene_gsd = None
        scene_shape = None
        
        if fecha_base.startswith("WV") or "worldview" in fecha_base.lower():
            if "MS" not in paths:
                logging.info(f"Saltando {fecha_base}: no hay escena MS disponible")
                continue
            if "PN" in paths:
                del paths["PN"]
        
        for key, path in paths.items():
            calib_dir = path / "Optical_Calibration"
            overview_file = calib_dir / "overview-trc.tif"
            if overview_file.exists():
                overview_path = overview_file
                break
        
        if not overview_path:
            for key, path in paths.items():
                calib_dir = path / "Optical_Calibration"
                tif_files = list(calib_dir.glob("*.tif"))
                if tif_files:
                    overview_path = tif_files[0]
                    break
        
        for key, path in paths.items():
            json_files = list(path.glob("*.json"))
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        item = json.load(f)
                    
                    if 'geometry' in item:
                        scene_geom = shape(item['geometry'])
                    elif 'bbox' in item:
                        scene_geom = box(*item['bbox'])
                    
                    scene_date = item.get('properties', {}).get('datetime')
                    if scene_date:
                        try:
                            scene_date = datetime.fromisoformat(scene_date.replace('Z', '+00:00'))
                        except:
                            scene_date = None
                    
                    properties = item.get('properties', {})
                    if 'proj:epsg' in properties:
                        scene_crs = f"EPSG:{properties['proj:epsg']}"
                    
                    if 'gsd' in properties:
                        scene_gsd = properties['gsd']
                    
                    if 'proj:shape' in properties:
                        scene_shape = properties['proj:shape']
                    
                    break
                except Exception as e:
                    logging.warning(f"Error leyendo JSON {json_files[0]}: {e}")
                    continue
        
        if scene_date is None:
            scene_date = parse_date_from_scene_id(fecha_base)
        
        if overview_path:
            scene = {
                'scene_id': fecha_base,
                'date': scene_date,
                'geometry': scene_geom,
                'overview_path': overview_path,
                'path': paths.get('MS') or paths.get('PN') or paths.get('OTRO'),
                'crs': scene_crs,
                'gsd': scene_gsd,
                'shape': scene_shape
            }
            scenes.append(scene)
            logging.info(f"Escena añadida: {fecha_base} (tipo: {list(paths.keys())})")
            if scene_crs:
                logging.info(f"  CRS: {scene_crs}")
            if scene_gsd:
                logging.info(f"  GSD: {scene_gsd:.6f}m")
            if scene_shape:
                logging.info(f"  Shape: {scene_shape}")
    
    return scenes


def find_scenes_for_aoi(aoi_geom, training_root):
    training_root = Path(training_root)
    collection_path = training_root / "collection.json"

    if not collection_path.exists():
        logging.warning(f"No se encontró collection.json en {training_root}. Usando escaneo de carpetas como fallback.")
        return find_scenes_for_aoi_legacy(aoi_geom, training_root)

    logging.info(f"Leyendo escenas desde {collection_path}")
    
    with open(collection_path, 'r', encoding='utf-8') as f:
        collection_data = json.load(f)

    all_item_data = []
    item_links = [link['href'] for link in collection_data.get('links', []) if link.get('rel') == 'item']
    for item_href in item_links:
        item_path = (collection_path.parent / item_href).resolve()
        if not item_path.exists():
            logging.warning(f"El archivo JSON del item no existe: {item_path}")
            continue
        try:
            with open(item_path, 'r', encoding='utf-8') as f:
                item_data = json.load(f)
                item_data['__item_path'] = item_path
                all_item_data.append(item_data)
        except Exception as e:
            logging.error(f"Error leyendo item JSON {item_path}: {e}")

    grouped_scenes = {}
    
    for item in all_item_data:
        scene_id = item.get('id', '')
        # Extraer nombre base para WorldView: usar el patrón común WV03N28_549166E087
        base_name = scene_id
        if 'WV03N28_549166E087' in scene_id:
            # Para WorldView, agrupar por el patrón común antes de los números de producto
            base_name = 'WV03N28_549166E087'
        else:
            # Para otros sensores, usar la lógica original
            if re.search(r"MS\d{2}", scene_id):
                base_name = scene_id.split('_MS')[0]
            elif re.search(r"PN\d{2}", scene_id):
                base_name = scene_id.split('_PN')[0]
            # Remover -calibrated si está presente
            if base_name.endswith('-calibrated'):
                base_name = base_name[:-11]
        
        if base_name not in grouped_scenes:
            grouped_scenes[base_name] = {}
        
        if re.search(r"MS\d{2}", scene_id):
            grouped_scenes[base_name]['MS'] = item
            logging.info(f"Agrupando escena MS: {scene_id} bajo base: {base_name}")
        elif re.search(r"PN\d{2}", scene_id):
            grouped_scenes[base_name]['PN'] = item
            logging.info(f"Agrupando escena PN: {scene_id} bajo base: {base_name}")
        else:
            grouped_scenes[base_name]['OTRO'] = item
            logging.info(f"Agrupando escena OTRO: {scene_id} bajo base: {base_name}")

    final_items_to_process = []
    for base_name, items in grouped_scenes.items():
        logging.info(f"Procesando grupo {base_name}: {list(items.keys())}")
        is_worldview = 'wv' in base_name.lower() or 'worldview' in base_name.lower()
        if is_worldview:
            if 'MS' in items:
                final_items_to_process.append(items['MS'])
                logging.info(f"Seleccionando escena MS: {items['MS']['id']}")
                if 'PN' in items:
                    logging.info(f"Ignorando escena panchromática {items['PN']['id']} porque se prioriza la multiespectral.")
            else:
                logging.warning(f"Grupo {base_name} no tiene escena MS; se omitirá para evitar PN.")
        else:
            final_items_to_process.extend(items.values())
    # El filtro ya se aplicó en la lógica anterior, no necesitamos duplicar

    scenes = []
    for item_data in final_items_to_process:
        try:
            scene_id = item_data.get('id')
            properties = item_data.get('properties', {})
            assets = item_data.get('assets', {})
            item_path = item_data['__item_path']

            scene_date_str = properties.get('datetime')
            scene_date = datetime.fromisoformat(scene_date_str.replace('Z', '+00:00')) if scene_date_str else parse_date_from_scene_id(scene_id)
            
            scene_geom = shape(item_data['geometry']) if 'geometry' in item_data else None
            scene_crs = f"EPSG:{properties['proj:epsg']}" if properties.get('proj:epsg') is not None else None
            scene_gsd = properties.get('gsd')
            scene_shape = properties.get('proj:shape')

            overview_asset = None
            if 'overview' in assets and 'href' in assets['overview']:
                overview_asset = assets['overview']
            else:
                for key, asset in assets.items():
                    if 'visual' in asset.get('roles', []) and 'href' in asset:
                        overview_asset = asset
                        logging.info(f"Usando asset '{key}' como overview para {scene_id}")
                        break
            
            if not overview_asset:
                logging.warning(f"No se encontró un asset visual/overview válido para {scene_id}. Saltando.")
                continue

            overview_path = (item_path.parent / overview_asset['href']).resolve()
            if not overview_path.exists():
                logging.warning(f"El archivo de imagen no existe: {overview_path}")
                continue
            
            scene_path = overview_path.parent.parent
            
            scene = {
                'scene_id': scene_id,
                'date': scene_date,
                'geometry': scene_geom,
                'overview_path': overview_path,
                'path': scene_path,
                'crs': scene_crs,
                'gsd': scene_gsd,
                'shape': scene_shape
            }
            scenes.append(scene)
            logging.info(f"Escena añadida: {scene_id} (Plataforma: {properties.get('platform', 'N/A')})")
            if scene_crs: logging.info(f"  CRS: {scene_crs}")
            if scene_gsd: logging.info(f"  GSD: {scene_gsd:.6f}m")
            if scene_shape: logging.info(f"  Shape: {scene_shape}")
        
        except Exception as e:
            import traceback
            logging.error(f"Error procesando datos para la escena {item_data.get('id', 'desconocida')}: {e}\n{traceback.format_exc()}")
            continue

    return scenes

def parse_date_from_scene_id(scene_id: str):
    """Intenta extraer una fecha (y opcionalmente hora) de un identificador de escena.
    Reconoce patrones AAAAMMDD o AAAAMMDDHHMMSS que empiecen por 20.
    Devuelve un objeto datetime o None si no encuentra nada útil."""
    # Buscar 14 dígitos (AAAAMMDDHHMMSS)
    m14 = re.search(r"(20\d{12})", scene_id)
    if m14:
        try:
            return datetime.strptime(m14.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    # Buscar 8 dígitos (AAAAMMDD)
    m8 = re.search(r"(20\d{6})", scene_id)
    if m8:
        try:
            return datetime.strptime(m8.group(1), "%Y%m%d")
        except ValueError:
            pass
    return None

def find_all_gpkg_files(data_root):
    training_annotations = Path(data_root) / "Training" / "Annotations"
    gpkg_files = []
    
    for root, _, files in os.walk(training_annotations):
        for file in files:
            if file.lower().endswith(('training.gpkg', 'testing.gpkg')):
                gpkg_files.append(Path(root) / file)
    
    return gpkg_files

def verify_spatial_coverage(scene, gpkg_path, layer_name, country=None):
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        
        scene_geom = scene.get('geometry')
        if scene_geom is None:
            logging.warning(f"No se pudo obtener geometría para la escena {scene['scene_id']}")
            return False
        
        # Obtener CRS de la escena (puede que la geometría esté en 4326 aunque el CRS declarado sea UTM)
        scene_crs = None
        if scene.get('crs'):
            scene_crs = scene['crs']
        else:
            scene_path = scene.get('path')
            if scene_path:
                json_files = list(Path(scene_path).glob("*.json"))
                if json_files:
                    try:
                        with open(json_files[0], 'r') as f:
                            item = json.load(f)
                        if 'properties' in item and 'proj:epsg' in item['properties']:
                            epsg_code = item['properties']['proj:epsg']
                            scene_crs = f"EPSG:{epsg_code}"
                            logging.info(f"CRS de escena obtenido del JSON: {scene_crs}")
                        else:
                            # Usar CRS específico del país si está disponible
                            if country and country in COUNTRY_TO_CRS:
                                scene_crs = COUNTRY_TO_CRS[country]
                                logging.info(f"CRS de escena no encontrado, usando CRS del país {country}: {scene_crs}")
                            else:
                                scene_crs = "EPSG:4326"
                                logging.info(f"CRS de escena no encontrado, usando default: {scene_crs}")
                    except:
                        if country and country in COUNTRY_TO_CRS:
                            scene_crs = COUNTRY_TO_CRS[country]
                            logging.info(f"Error leyendo JSON, usando CRS del país {country}: {scene_crs}")
                        else:
                            scene_crs = "EPSG:4326"
                            logging.info(f"Error leyendo JSON, usando CRS default: {scene_crs}")
                else:
                    if country and country in COUNTRY_TO_CRS:
                        scene_crs = COUNTRY_TO_CRS[country]
                        logging.info(f"No se encontró JSON, usando CRS del país {country}: {scene_crs}")
                    else:
                        scene_crs = "EPSG:4326"
                        logging.info(f"No se encontró JSON, usando CRS default: {scene_crs}")
            else:
                if country and country in COUNTRY_TO_CRS:
                    scene_crs = COUNTRY_TO_CRS[country]
                    logging.info(f"No se pudo determinar CRS de escena, usando CRS del país {country}: {scene_crs}")
                else:
                    scene_crs = "EPSG:4326"
                    logging.info(f"No se pudo determinar CRS de escena, usando default: {scene_crs}")
        
        # Verificar que el CRS sea válido
        if scene_crs is None or str(scene_crs) == 'EPSG:None':
            if country and country in COUNTRY_TO_CRS:
                scene_crs = COUNTRY_TO_CRS[country]
                logging.warning(f"CRS inválido detectado, usando CRS del país {country}: {scene_crs}")
            else:
                scene_crs = "EPSG:4326"
                logging.warning(f"CRS inválido detectado, usando fallback: {scene_crs}")
        
        # --- Asegurar coherencia de CRS entre geometrías ---
        # Si la geometría de la escena parece estar en grados (lon/lat) pero el CRS es proyectado,
        # reproyectamos la geometría a ese CRS. Criterio: valores |x|<180 and |y|<90 → grados.

        if scene_crs and scene_crs != "EPSG:4326":
            minx, miny, maxx, maxy = scene_geom.bounds
            if abs(minx) <= 180 and abs(maxx) <= 180 and abs(miny) <= 90 and abs(maxy) <= 90:
                # Parece estar en grados → reproyectar al CRS declarado
                try:
                    transformer = Transformer.from_crs("EPSG:4326", scene_crs, always_xy=True)
                    scene_geom = shapely_transform(transformer.transform, scene_geom)
                    logging.info("Reproyectada geometría de escena de EPSG:4326 → %s", scene_crs)
                except Exception as e:
                    logging.warning("No se pudo reproyectar geometría de escena: %s", e)

        scene_gdf = gpd.GeoDataFrame(geometry=[scene_geom], crs=scene_crs)

        # Reproyectar polígonos de GPKG al CRS de la escena para comparar correctamente
        if scene_crs and gdf.crs and gdf.crs != scene_crs:
            try:
                gdf = gdf.to_crs(scene_crs)
                logging.info("Reproyectado GPKG de %s → %s para comparación", gdf.crs, scene_crs)
            except Exception as e:
                logging.warning("No se pudo reproyectar GPKG: %s", e)
        
        polygons_in_scene = 0
        total_polygons = len(gdf)
        
        for geom in gdf.geometry:
            if geom.intersects(scene_gdf.geometry.iloc[0]):
                polygons_in_scene += 1
        
        coverage_percentage = (polygons_in_scene / total_polygons) * 100
        
        logging.info(f"Cobertura espacial para {scene['scene_id']}:")
        logging.info(f"  Polígonos en escena: {polygons_in_scene}/{total_polygons} ({coverage_percentage:.1f}%)")
        logging.info(f"  Bounds escena: {scene_gdf.total_bounds}")
        logging.info(f"  Bounds polígonos: {gdf.total_bounds}")
        
        if coverage_percentage < 10:
            logging.warning(f"⚠️ BAJA COBERTURA: Solo {coverage_percentage:.1f}% de polígonos están en esta escena")
            return False
        elif coverage_percentage > 90:
            logging.info(f"✅ ALTA COBERTURA: {coverage_percentage:.1f}% de polígonos están en esta escena")
            return True
        else:
            logging.info(f"⚠️ COBERTURA MODERADA: {coverage_percentage:.1f}% de polígonos están en esta escena")
            return True
            
    except Exception as e:
        logging.error(f"Error verificando cobertura espacial: {e}")
        return False

def clean_and_validate_geometries(gdf):
    logging.info(f"Limpiando y validando {len(gdf)} geometrías...")
    
    # Usar la nueva sintaxis recomendada para evitar warnings
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    logging.info(f"Después de filtrar vacías/nulas: {len(gdf)} geometrías")
    
    valid_geometries = []
    invalid_count = 0
    
    for idx, geom in enumerate(gdf.geometry):
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
                invalid_count += 1
            
            if geom.is_valid and not geom.is_empty:
                valid_geometries.append(geom)
            else:
                logging.warning(f"Geometría {idx} no se pudo reparar")
        except Exception as e:
            logging.warning(f"Error procesando geometría {idx}: {e}")
    
    logging.info(f"Geometrías reparadas: {invalid_count}")
    logging.info(f"Geometrías válidas finales: {len(valid_geometries)}")
    
    if valid_geometries:
        gdf_clean = gpd.GeoDataFrame(geometry=valid_geometries, crs=gdf.crs)
        return gdf_clean
    else:
        logging.error("No quedaron geometrías válidas después de la limpieza")
        return None

def get_aoi_geometry(gdf):
    try:
        aoi_geom = gdf.to_crs(4326).geometry.union_all()
        return aoi_geom
    except Exception as e:
        logging.warning(f"Error en union_all original: {e}")
        logging.info("Intentando limpiar geometrías...")
        
        gdf_clean = clean_and_validate_geometries(gdf)
        if gdf_clean is None:
            logging.error("No se pudo obtener área de interés válida")
            return None
        
        try:
            aoi_geom = gdf_clean.to_crs(4326).geometry.union_all()
            return aoi_geom
        except Exception as e2:
            logging.error(f"Error en union_all con geometrías limpias: {e2}")
            
            logging.info("Usando bounding box como área de interés")
            try:
                bbox = gdf_clean.to_crs(4326).total_bounds
                from shapely.geometry import box
                aoi_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
                return aoi_geom
            except Exception as e3:
                logging.error(f"Error creando bounding box: {e3}")
                return None

def extract_city_from_gpkg(gpkg_path):
    """
    Extrae la ciudad del nombre del archivo GPKG.
    Ejemplo: 'Gaziantep_TRAINING.gpkg' -> 'Gaziantep'
             'marash_training.gpkg' -> 'marash'
             'Antakya_TRAINING-Extent.gpkg' -> 'Antakya'
    """
    name = gpkg_path.name
    # Quitar extensión
    name = name.replace('.gpkg', '')
    # Separar por guion bajo o por guion
    city = re.split(r'[_-]', name)[0]
    return city.capitalize()

def calculate_buffer_meters(scene_id, gsd=None, user_buffer=None):
    """
    Calcula el buffer en metros según el tipo de sensor y GSD.
    
    Args:
        scene_id: ID de la escena para detectar el sensor
        gsd: Ground Sample Distance en metros
        user_buffer: Buffer especificado por el usuario (opcional)
    
    Returns:
        Buffer en metros
    """
    if user_buffer is not None:
        return user_buffer
    
    scene_id_upper = scene_id.upper()
    
    # Detectar tipo de sensor
    if any(sensor in scene_id_upper for sensor in ['DS_PHR', 'PLEIADES']):
        return 20.0  # Pleiades
    elif any(sensor in scene_id_upper for sensor in ['WV', 'WORLDVIEW', 'GE', 'GEOEYE']):
        return 40.0  # WorldView/GeoEye
    else:
        return 60.0  # Otros sensores

def visualize_scene_with_polygons(scene, gpkg_path, layer_name, country, city=None, show_polygons=True, filter_outside_polygons=True):
    try:
        has_coverage = verify_spatial_coverage(scene, gpkg_path, layer_name, country)
        if not has_coverage:
            logging.warning(f"Los polígonos no están en el área de cobertura de {scene['scene_id']}")
        
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        gdf_clean = clean_and_validate_geometries(gdf)
        if gdf_clean is None:
            logging.error(f"No se pudieron cargar geometrías válidas para {gpkg_path.name}")
            return
        
        # --- OPCIÓN 2: Chequeo y forzado de CRS ---
        with rasterio.open(scene['overview_path']) as src:
            img_crs = src.crs
            logging.info(f"CRS de la imagen: {img_crs}")
            logging.info(f"CRS de los polígonos antes de reproyección: {gdf_clean.crs}")
            if gdf_clean.crs != img_crs:
                try:
                    gdf_clean = gdf_clean.to_crs(img_crs)
                    logging.info(f"Polígonos reproyectados a CRS de la imagen: {img_crs}")
                except Exception as e:
                    logging.error(f"Error reproyectando GPKG a CRS de la imagen: {e}")
                    return
            else:
                logging.info("No es necesario reproyectar los polígonos.")

            # --- Comprobar si la matriz Affine incluye rotación ---
            has_rotation = not (abs(src.transform.b) < 1e-9 and abs(src.transform.d) < 1e-9)

            if has_rotation:
                logging.info("La imagen tiene rotación; no se puede componer RGB fácilmente. Se usará overview.")
                img = src.read(1, out_shape=(src.height // 4, src.width // 4))
                img_transform = src.transform * src.transform.scale(src.width / (src.width // 4), src.height / (src.height // 4))
                img = _normalize_for_display(img)

            else:
                scale_factor = 4
                out_shape = (src.height // scale_factor, src.width // scale_factor)
                
                # Intentar leer bandas RGB separadas
                with open(scene['path'] / f"{scene['scene_id']}.json", 'r') as f:
                    item_data = json.load(f)
                
                img, img_transform = _read_rgb_bands(item_data.get('assets', {}), scene['path'], out_shape)

                # Si falla la lectura RGB, usar el overview como fallback
                if img is None:
                    num_bands = min(3, src.count)
                    data = src.read(list(range(1, num_bands + 1)), out_shape=out_shape)
                    img = np.moveaxis(data, 0, -1)
                    img_transform = src.transform * src.transform.scale(src.width / out_shape[1], src.height / out_shape[0])

                img = _normalize_for_display(img)
                img_bounds = src.bounds
            
            # Si reproyectamos la imagen podemos necesitar actualizar extent con transform.
            if has_rotation:
                left, bottom, right, top = img_bounds
            else:
                left, bottom, right, top = img_bounds[0], img_bounds[1], img_bounds[2], img_bounds[3]
            
            max_polys = 50
            fig, ax = plt.subplots(figsize=(12, 8))
            
            left, bottom, right, top = img_bounds[0], img_bounds[1], img_bounds[2], img_bounds[3]

            ax.imshow(img, extent=(left, right, bottom, top))

            patches = []
            shown = 0
            centroids = []
            
            # --- OPCIÓN 4: Limitar el número de polígonos a mostrar ---
            max_polys = 100
            if show_polygons and len(gdf_clean) > max_polys:
                logging.info(f"Demasiados polígonos ({len(gdf_clean)}), mostrando solo {max_polys} aleatorios para evitar sobrecarga visual.")
                gdf_clean = gdf_clean.sample(max_polys, random_state=42)
            
            # --- APLICAR BUFFER A LOS POLÍGONOS ---
            if show_polygons and not gdf_clean.empty:
                # Calcular buffer automático basado en el sensor
                scene_gsd = scene.get('gsd', None)
                auto_buffer = calculate_buffer_meters(scene['scene_id'], scene_gsd)
                logging.info(f"Aplicando buffer de {auto_buffer}m a los polígonos (sensor detectado en {scene['scene_id']})")
                
                # Si estamos en CRS geográfico, reproyectar temporalmente para el buffer
                original_crs = gdf_clean.crs
                if gdf_clean.crs.is_geographic:
                    # Buscar un CRS proyectado apropiado basado en el centroide
                    centroid = gdf_clean.geometry.union_all().centroid
                    lon, lat = centroid.x, centroid.y
                    
                    # Estimar zona UTM
                    utm_zone = int((lon + 180) / 6) + 1
                    utm_crs = f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"
                    
                    logging.info(f"Reproyectando a {utm_crs} para aplicar buffer en metros correctamente")
                    gdf_clean = gdf_clean.to_crs(utm_crs)
                    gdf_clean.geometry = gdf_clean.geometry.buffer(auto_buffer)
                    gdf_clean = gdf_clean.to_crs(original_crs)
                    logging.info(f"Buffer aplicado y reproyectado de vuelta a {original_crs}")
                else:
                    # Ya está en CRS proyectado, aplicar buffer directamente
                    gdf_clean.geometry = gdf_clean.geometry.buffer(auto_buffer)
            
            # --- FILTRAR POLÍGONOS DENTRO DE LOS LÍMITES DE LA IMAGEN ---
            if show_polygons and not gdf_clean.empty and filter_outside_polygons:
                from shapely.geometry import box
                img_bbox = box(left, bottom, right, top)
                
                original_count = len(gdf_clean)
                # Filtrar solo polígonos que intersectan con la imagen
                gdf_clean = gdf_clean[gdf_clean.geometry.intersects(img_bbox)]
                filtered_count = len(gdf_clean)
                logging.info(f"Polígonos filtrados: {original_count} → {filtered_count} (eliminados {original_count - filtered_count} fuera de imagen)")
                
                if gdf_clean.empty:
                    logging.warning("No hay polígonos dentro de los límites de la imagen")
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
                                try:
                                    patches.append(Polygon(list(poly.exterior.coords), closed=True, fill=False, edgecolor='red', linewidth=1.5))
                                    centroids.append(poly.centroid)
                                    shown += 1
                                except Exception as e:
                                    logging.warning(f"Error procesando parte de MultiPolygon: {e}")
                                    continue
                    except Exception as e:
                        logging.warning(f"Error procesando geometría para visualización: {e}")
                        continue
                
                if patches:
                    ax.add_collection(PatchCollection(patches, match_original=True))
                    
                    for idx, c in enumerate(centroids):
                        ax.plot(c.x, c.y, 'o', color='red', markersize=8)
                        ax.text(c.x, c.y, str(idx+1), color='yellow', fontsize=8, fontweight='bold', ha='center', va='center')
                else:
                    logging.warning("No se pudieron dibujar polígonos válidos")
            else:
                ax.text(0.5, 0.95, 'ESCENA PRE-TERREMOTO\nLos polígonos de daños no deberían aparecer aquí', 
                       transform=ax.transAxes, fontsize=14, ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            date_str = scene['date'].strftime('%Y-%m-%d %H:%M') if scene['date'] else 'Fecha desconocida'
            img_epsg = src.crs.to_epsg() if src.crs else 'No definido'
            poly_epsg = gdf.crs.to_epsg() if gdf.crs else 'No definido'
            coverage_status = "OK" if has_coverage else "WARNING"
            polygon_status = "MOSTRADOS" if show_polygons else "OCULTOS (PRE-TERREMOTO)"
            
            # Determinar si es PRE o POST terremoto
            pre_post_status = "DESCONOCIDO"
            pre_post_color = "gray"
            if scene['date']:
                # Fechas aproximadas de los terremotos principales en el dataset
                earthquake_dates = {
                    "Turkey": datetime(2023, 2, 6),
                    "Syria": datetime(2023, 2, 6), 
                    "Morocco": datetime(2023, 9, 8),
                    "Afghanistan": datetime(2023, 10, 7),
                    "China": datetime(2025, 1, 7),
                    "Myanmar": datetime(2025, 3, 27)
                }
                
                if country in earthquake_dates:
                    earthquake_date = earthquake_dates[country]
                    # Normalizar fechas para comparación (remover zona horaria si es necesario)
                    scene_date_naive = scene['date'].replace(tzinfo=None) if scene['date'].tzinfo else scene['date']
                    earthquake_date_naive = earthquake_date.replace(tzinfo=None) if earthquake_date.tzinfo else earthquake_date
                    
                    if scene_date_naive < earthquake_date_naive:
                        pre_post_status = "PRE-TERREMOTO"
                        pre_post_color = "orange"
                    else:
                        pre_post_status = "POST-TERREMOTO"
                        pre_post_color = "red"
            
            # Agregar ciudad al título
            city_line = f"Ciudad: {city}" if city else ""
            title = f"País: {country}\n{city_line}\nEscena: {scene['scene_id']}\nFecha: {date_str}\nCRS Imagen: EPSG:{img_epsg} | CRS Polígonos: EPSG:{poly_epsg}\nCobertura: {coverage_status} | Polígonos: {polygon_status}"
            plt.title(title, fontsize=10)
            
            # Añadir indicador PRE/POST en una esquina prominente
            ax.text(0.02, 0.98, pre_post_status, transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=pre_post_color, alpha=0.8, edgecolor='black'),
                   color='white')
            
            plt.tight_layout()
            
            plt.show()
            logging.info("Figura mostrada en pantalla")
            
    except Exception as e:
        logging.error(f"Error visualizando escena {scene['scene_id']}: {e}")

def visualize_pre_post_scenes_with_polygons(pre_scene, post_scene, gpkg_path, layer_name, country, city=None, show_polygons=True, filter_outside_polygons=True):
    """
    Visualiza lado a lado la escena pre y post-terremoto con los polígonos superpuestos.
    """
    import copy
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    gdf_clean = clean_and_validate_geometries(gdf)
    if gdf_clean is None:
        logging.error(f"No se pudieron cargar geometrías válidas para {gpkg_path.name}")
        return

    scenes = [pre_scene, post_scene]
    titles = ["PRE-TERREMOTO", "POST-TERREMOTO"]
    colors = ["orange", "red"]
    imgs = []
    img_extents = []
    img_crss = []
    gdfs = []
    scale_factor = 4
    for idx, scene in enumerate(scenes):
        with rasterio.open(scene['overview_path']) as src:
            img_crs = src.crs
            img_crss.append(img_crs)
            gdf_proj = gdf_clean
            if gdf_clean.crs != img_crs:
                try:
                    gdf_proj = gdf_clean.to_crs(img_crs)
                except Exception as e:
                    logging.error(f"Error reproyectando GPKG a CRS de la imagen: {e}")
                    return
            has_rotation = not (abs(src.transform.b) < 1e-9 and abs(src.transform.d) < 1e-9)
            if has_rotation:
                img = src.read(1, out_shape=(src.height // scale_factor, src.width // scale_factor))
                img = _normalize_for_display(img)
                img_bounds = src.bounds
            else:
                out_shape = (src.height // scale_factor, src.width // scale_factor)
                with open(scene['path'] / f"{scene['scene_id']}.json", 'r') as f:
                    item_data = json.load(f)
                img, _ = _read_rgb_bands(item_data.get('assets', {}), scene['path'], out_shape)
                if img is None:
                    num_bands = min(3, src.count)
                    data = src.read(list(range(1, num_bands + 1)), out_shape=out_shape)
                    img = np.moveaxis(data, 0, -1)
                img = _normalize_for_display(img)
                img_bounds = src.bounds
            imgs.append(img)
            img_extents.append((img_bounds[0], img_bounds[2], img_bounds[1], img_bounds[3]))
            gdfs.append(gdf_proj)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for idx, ax in enumerate(axes):
        img = imgs[idx]
        left, right, bottom, top = img_extents[idx]
        ax.imshow(img, extent=(left, right, bottom, top))
        gdf_proj = gdfs[idx]
        # Polígonos
        patches = []
        centroids = []
        max_polys = 100
        gdf_show = gdf_proj
        if show_polygons and len(gdf_proj) > max_polys:
            gdf_show = gdf_proj.sample(max_polys, random_state=42)
        # Buffer
        if show_polygons and not gdf_show.empty:
            scene_gsd = scenes[idx].get('gsd', None)
            auto_buffer = calculate_buffer_meters(scenes[idx]['scene_id'], scene_gsd)
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
        # Filtrar fuera de imagen
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
        ax.set_title(f"{titles[idx]}\nPaís: {country}\n{city_line}\nEscena: {scenes[idx]['scene_id']}\nFecha: {date_str}\nCRS Imagen: EPSG:{img_epsg} | CRS Polígonos: EPSG:{poly_epsg}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    logging.info("Figura pre/post mostrada en pantalla")







def main():
    parser = argparse.ArgumentParser(description="Visualizador de escenas TIFF con polígonos de anotación.")
    
    parser.add_argument('--data-root', type=str, 
                        default="data/international-charter-eo4ai-etq-challenge",
                        help='Ruta raíz de los datos.')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limitar número de GPKGs a procesar.')
    parser.add_argument('--country', type=str, default=None,
                        help='Filtrar y procesar solo escenas de este país (ej: Turkey, China, Afghanistan, etc).')
    parser.add_argument('--filter-outside-polygons', action='store_true', default=True,
                        help='Filtrar polígonos que caen fuera de los límites de la imagen.')

    
    args = parser.parse_args()

    # Encontrar y filtrar archivos GPKG
    gpkg_files = find_all_gpkg_files(args.data_root)
    logging.info(f"Encontrados {len(gpkg_files)} archivos GPKG")
    
    if args.limit:
        gpkg_files = gpkg_files[:args.limit]
        logging.info(f"Limitado a {len(gpkg_files)} archivos GPKG")

    if args.country:
        country_filter = args.country.strip().lower()
        valid_countries = set([v.lower() for v in EVENT_TO_COUNTRY.values()])
        if country_filter not in valid_countries:
            logging.error(f"País '{args.country}' no reconocido. Opciones válidas: {sorted(EVENT_TO_COUNTRY.values())}")
            return
        gpkg_files = [gpkg for gpkg in gpkg_files 
                     if get_country_from_event(gpkg.parent.name).lower() == country_filter]
        logging.info(f"Filtrado por país '{args.country}': {len(gpkg_files)} archivos GPKG")
        if not gpkg_files:
            logging.warning(f"No se encontraron archivos GPKG para el país '{args.country}'")
            return
    
    for gpkg_path in gpkg_files:
        logging.info(f"\nProcesando GPKG: {gpkg_path.name}")
        
        # Cargar datos del GPKG
        with fiona.Env():
            with fiona.open(gpkg_path) as gpkg:
                layer_name = gpkg.name
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        logging.info(f"Polígonos cargados: {len(gdf)}")
        
        # Obtener geometría del área de interés
        aoi_geom = get_aoi_geometry(gdf)
        if aoi_geom is None:
            logging.warning(f"No se pudo obtener un área de interés válida para {gpkg_path.name}. Saltando escenas.")
            continue
        
        # Encontrar escenas
        event_name = gpkg_path.parent.name
        event_dir = Path(args.data_root) / "Training" / event_name
        country = get_country_from_event(event_name)
        logging.info(f"País: {country}")
        
        if not event_dir.exists():
            logging.warning(f"Directorio del evento no encontrado: {event_dir}")
            continue
        
        scenes = find_scenes_for_aoi(aoi_geom, event_dir)
        logging.info(f"Escenas encontradas: {len(scenes)}")
        
        if not scenes:
            logging.warning(f"No se encontraron escenas para {event_name}")
            continue
        
        # Verificar cobertura espacial
        scenes_with_coverage = [scene for scene in scenes 
                              if verify_spatial_coverage(scene, gpkg_path, layer_name, country)]
        if not scenes_with_coverage:
            logging.warning(f"No se encontraron escenas con cobertura de polígonos para {event_name}")
            scenes_with_coverage = scenes
        

        
        logging.info(f"Escenas con cobertura de polígonos: {len(scenes_with_coverage)}/{len(scenes)}")
        
        # Visualización pre/post lado a lado
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
        
        # Visualización individual
        if len(scenes) == 1:
            scenes_with_coverage = scenes
        

        
        # Visualizar escenas individuales
        for scene in scenes_with_coverage:
            logging.info(f"Visualizando escena: {scene['scene_id']}")
            city = extract_city_from_gpkg(gpkg_path)
            
            # Determinar si mostrar polígonos
            show_polygons = True
            if len(scenes) >= 2:
                dates = [s['date'] for s in scenes if s['date']]
                if dates:
                    scene_date_naive = scene['date'].replace(tzinfo=None) if scene['date'].tzinfo else scene['date']
                    if scene_date_naive == min(dates):  # Pre-terremoto
                        show_polygons = True
                        status = "PRE-TERREMOTO"
                    elif scene_date_naive == max(dates):  # Post-terremoto
                        show_polygons = True
                        status = "POST-TERREMOTO"
                    else:  # Intermedia
                        show_polygons = verify_spatial_coverage(scene, gpkg_path, layer_name, country)
                        status = "INTERMEDIA"
                else:
                    show_polygons = verify_spatial_coverage(scene, gpkg_path, layer_name, country)
                    status = "ÚNICA"
            else:
                show_polygons = verify_spatial_coverage(scene, gpkg_path, layer_name, country)
                status = "ÚNICA"
            
            logging.info(f"Visualizando escena {status}: {scene['scene_id']}")
            visualize_scene_with_polygons(scene, gpkg_path, layer_name, country, city, 
                                        show_polygons=show_polygons, filter_outside_polygons=args.filter_outside_polygons)
        
        logging.info(f"Completado GPKG: {gpkg_path.name}. Continuando con el siguiente...")

if __name__ == "__main__":
    main() 