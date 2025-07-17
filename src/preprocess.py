"""
Earthquake Building Damage Preprocessing Pipeline
------------------------------------------------
Prepares the "international-charter-eo4ai-etq-challenge" dataset for ML/GeoAI competitions.

Features:
- Reads STAC-style catalogs, pansharpens, coregisters, and tiles images.
- Supports multi-class (damage_grade) or binary masks (with --binary).
- Fast rasterization, memory-efficient tile streaming, multiprocessing.
- Balanced splits: full (100%), val (~10% cities), small (~10% tiles/city).
- Computes global mean/std (stats.json). Optionally saves debug overlays.
- CLI: --val_pct --small_pct --binary --workers --debug_vis N

Example usage:
python preprocess.py --raw_data_root ./Training --output_root ./data/preprocessed --workers 4 --debug_vis 16
"""
import json
import hashlib
from pathlib import Path
import torch
import numpy as np
import rasterio
import geopandas as gpd
from tqdm import tqdm
from tqdm import trange
from skimage.registration import phase_cross_correlation
from skimage.transform import warp
import rasterio.features
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.ndimage import shift as ndi_shift
import glob
import re

RAW_DATA_ROOT = Path("data/international-charter-eo4ai-etq-challenge/Training")
OUTPUT_ROOT = Path("data/preprocessed")
TILE_SIZE = 512
TILE_OVERLAP = 64

# --- read_stac adaptado a la estructura real y pre/post evento ---
def read_stac(raw_data_root: Path, annotations_root: Path = None):
    annotations_root = annotations_root or (raw_data_root / "Annotations")
    tqdm.write(f"[INFO] Buscando ciudades en {raw_data_root} ...")
    for city_dir in raw_data_root.iterdir():
        if not city_dir.is_dir() or not city_dir.name.startswith("Call_"):
            continue
        collection_path = city_dir / "collection.json"
        if not collection_path.exists():
            tqdm.write(f"[WARN] No collection.json en {city_dir}, se omite.")
            continue
        tqdm.write(f"[INFO] Procesando ciudad: {city_dir.name}")
        with open(collection_path) as f:
            collection = json.load(f)
        # --- Extraer fecha del evento del campo de descripción ---
        desc = collection.get("description", "")
        m = re.search(r"Date of Charter Activation: (\d{4}-\d{2}-\d{2})", desc)
        if not m:
            tqdm.write(f"[WARN] No se encontró fecha de evento en descripción de {city_dir.name}, se omite.")
            continue
        event_date = m.group(1)
        event_date = event_date + "T00:00:00Z"
        # Buscar archivos de anotaciones para la ciudad
        ann_city_dir = annotations_root / city_dir.name
        gpkg_files = sorted(ann_city_dir.glob("*.gpkg")) if ann_city_dir.exists() else []
        if not gpkg_files:
            tqdm.write(f"[WARN] No se encontró archivo de anotaciones para {city_dir.name}, se omite.")
            continue
        if len(gpkg_files) > 1:
            tqdm.write(f"[WARN] Varios archivos de anotaciones en {ann_city_dir}, se procesarán todos.")
        # Iterar sobre cada archivo de anotaciones
        for label_path in gpkg_files:
            tqdm.write(f"[INFO] Usando archivo de anotaciones: {label_path.name}")
            # --- Clasificar escenas pre y post evento ---
            pre_items = []
            post_items = []
            for link in collection.get("links", []):
                if link.get("rel") != "item":
                    continue
                scene_json = city_dir / link["href"]
                if not scene_json.exists():
                    tqdm.write(f"[WARN] No existe escena {scene_json}, se omite.")
                    continue
                with open(scene_json) as sf:
                    scene = json.load(sf)
                scene_date = scene.get("properties", {}).get("datetime", None)
                if not scene_date:
                    # Intentar extraer fecha del nombre
                    m2 = re.search(r"(\d{8})", scene_json.name)
                    if m2:
                        scene_date = m2.group(1)[:4] + "-" + m2.group(1)[4:6] + "-" + m2.group(1)[6:8] + "T00:00:00Z"
                if not scene_date:
                    tqdm.write(f"[WARN] No se encontró fecha para escena {scene_json}, se omite.")
                    continue
                if scene_date < event_date:
                    pre_items.append(scene)
                else:
                    post_items.append(scene)
            if not pre_items or not post_items:
                tqdm.write(f"[WARN] Faltan escenas pre o post evento en {city_dir.name}, se omite.")
                continue
            # Tomar la primera pre y la primera post (puedes mejorar esto si hay varias)
            pre_scene = pre_items[0]
            post_scene = post_items[0]
            def extract_img_paths(scene):
                assets = scene.get("assets", {})
                return {k: (city_dir / assets[k]["href"]).resolve() for k in assets if "href" in assets[k]}
            img_paths = {
                "pre": extract_img_paths(pre_scene),
                "post": extract_img_paths(post_scene)
            }
            # Extraer info de pre y post
            pre_info = {
                "scene_id": pre_scene.get("id", "pre"),
                "date": pre_scene.get("properties", {}).get("datetime", None),
                "type": "pre"
            }
            post_info = {
                "scene_id": post_scene.get("id", "post"),
                "date": post_scene.get("properties", {}).get("datetime", None),
                "type": "post"
            }
            images_info = {"pre": pre_info, "post": post_info}
            yield {
                "city": city_dir.name,
                "id": f"{pre_scene.get('id', 'pre')}_{post_scene.get('id', 'post')}__{label_path.stem}",
                "img_paths": img_paths,  # dict: pre/post -> band -> path
                "label_path": str(label_path),
                "images_info": images_info, # Añadir 'images_info' a cada entrada
            }

# --- pansharpen_if_pan ---
def pansharpen_if_pan(ms_img, pan_img=None):
    if pan_img is None:
        return ms_img
    ms_sum = ms_img.sum(axis=0, keepdims=True)
    ms_sum[ms_sum == 0] = 1
    brovey = ms_img * (pan_img / ms_sum)
    return brovey

# --- coregister ultra-rápido (solo traslación) ---
def coregister(pre, post):
    pre_mean = pre.mean(0)
    post_mean = post.mean(0)
    if np.all(pre_mean == 0) or np.all(post_mean == 0):
        # No hacer registro si alguna imagen está vacía
        return post
    shift, error, diffphase = phase_cross_correlation(pre_mean, post_mean, upsample_factor=10)
    dy = float(np.atleast_1d(shift[0]))
    dx = float(np.atleast_1d(shift[1]))
    return ndi_shift(post, shift=(0, dy, dx), mode="nearest")

# --- rasterize_labels (usa cache de GPKG) ---
gdf_cache = {}
def get_gdf(gpkg_path):
    if gpkg_path not in gdf_cache:
        gdf_cache[gpkg_path] = gpd.read_file(gpkg_path)
    return gdf_cache[gpkg_path]

def rasterize_labels(gpkg_path: Path, out_shape, transform, binary=False):
    gdf = get_gdf(gpkg_path)
    # Filtrar geometrías vacías y nulas
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)
    if "damaged" in gdf.columns:
        # 0: no dañado, 1: dañado, NaN: a predecir (usamos 255 para marcar estos píxeles)
        values = []
        for row in gdf.itertuples():
            val = getattr(row, "damaged", None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                values.append(255)  # Clase especial para "a predecir"
            else:
                values.append(int(val))
    else:
        values = [1 for _ in gdf.geometry]
    burned = rasterio.features.rasterize(
        [(geom, v) for geom, v in zip(gdf.geometry, values)],
        out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8
    )
    if binary:
        # Solo 0 y 1, pero mantenemos 255 para "a predecir"
        mask = (burned == 1).astype(np.uint8)
        mask[burned == 255] = 255
        burned = mask
    return burned

# --- save_tensor ---
def save_tensor(tensor, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_path, _use_new_zipfile_serialization=True)

# --- deterministic splits ---
def is_in_small_split(tile_id, pct=0.1):
    return int(hashlib.md5(tile_id.encode()).hexdigest(), 16) % int(1/pct) == 0

def is_val_city(city, pct=0.1):
    return int(hashlib.md5(city.encode()).hexdigest(), 16) % int(1/pct) == 0

# --- OnlineStats ultra-rápido y mergeable ---
class OnlineStats:
    def __init__(self, c):
        self.count = 0
        self.sum   = np.zeros(c, dtype=np.float64)
        self.sumsq = np.zeros(c, dtype=np.float64)

    def update(self, x):  # x: (C, H, W)
        n_channels = x.shape[0]
        n_pix = x.shape[1] * x.shape[2]
        self.count += n_pix
        self.sum   += x.reshape(n_channels, -1).sum(axis=1)
        self.sumsq += (x**2).reshape(n_channels, -1).sum(axis=1)

    def merge(self, other):
        self.count += other.count
        self.sum   += other.sum
        self.sumsq += other.sumsq

    def finalize(self):
        mean = self.sum / self.count
        var  = self.sumsq / self.count - mean**2
        std  = np.sqrt(np.maximum(var, 0))
        return mean.tolist(), std.tolist()

# --- debug visualization ---
def save_debug_vis(tile_img, tile_mask, out_path):
    c, h, w = tile_img.shape
    img = tile_img[:3]  # RGB or first 3 bands
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = np.moveaxis(img, 0, -1)
    mask = tile_mask
    plt.figure(figsize=(3,3))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4, cmap='jet')
    plt.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# --- Utilidad para seleccionar imágenes pre/post y bandas, tile por tile ---
def get_tile_from_bands(img_paths, y, x, tile_size):
    # img_paths: {"pre": {...}, "post": {...}}
    def read_bands(band_paths):
        ms_bands = [b for b in ["blue", "green", "red", "nir"] if b in band_paths]
        pan_band = "pan" if "pan" in band_paths else None
        tile_ms = None
        tile_pan = None
        if ms_bands:
            tile_ms = []
            for b in ms_bands:
                with rasterio.open(str(band_paths[b])) as src:
                    tile = src.read(1, window=rasterio.windows.Window(x, y, tile_size, tile_size)).astype(np.float32)
                    tile_ms.append(tile)
            tile_ms = np.stack(tile_ms, axis=0)
        if pan_band:
            with rasterio.open(str(band_paths[pan_band])) as src:
                tile_pan = src.read(1, window=rasterio.windows.Window(x, y, tile_size, tile_size)).astype(np.float32)
        return tile_ms, tile_pan
    pre_ms, pre_pan = read_bands(img_paths["pre"])
    post_ms, post_pan = read_bands(img_paths["post"])
    return pre_ms, pre_pan, post_ms, post_pan

# --- per-city processing (fusionada con opción de contar tiles mixtos) ---
def process_city(args, count_mixed_tiles=False):
    (city, items, params) = args
    tqdm.write(f"[INFO] Iniciando procesamiento de ciudad: {city} ({len(items)} escenas)")
    small_split = params['small_split']
    binary = params['binary']
    debug_vis = params['debug_vis']
    output_root = params['output_root']
    tile_size = params['tile_size']
    tile_overlap = params['tile_overlap']
    stats = None
    meta = {s: [] for s in ['full', 'small', 'test']}
    debug_samples = []
    import gc
    total_tiles = 0
    ignored_mixed_tiles = 0
    for item in items:
        id_ = item["id"]
        img_paths = item["img_paths"]
        label_path = item["label_path"]
        with rasterio.open(str(list(img_paths["pre"].values())[0])) as src:
            H, W = src.height, src.width
            pre_profile = src.profile
            transform = src.transform
        mask = rasterize_labels(label_path, (H, W), transform, binary=binary)
        step = tile_size - tile_overlap
        y_starts = list(range(0, H - tile_size + 1, step))
        x_starts = list(range(0, W - tile_size + 1, step))
        if not y_starts or y_starts[-1] != H - tile_size:
            y_starts.append(H - tile_size)
        if not x_starts or x_starts[-1] != W - tile_size:
            x_starts.append(W - tile_size)
        num_tiles_item = len(y_starts) * len(x_starts)
        tile_counter = 0
        for y in y_starts:
            for x in x_starts:
                tile_counter += 1
                total_tiles += 1
                if tile_counter % max(1, num_tiles_item // 10) == 0 or tile_counter == num_tiles_item:
                    percent = 100 * tile_counter / num_tiles_item
                    tqdm.write(f"[PROGRESS] Ciudad: {city}, Escena: {id_}, {tile_counter}/{num_tiles_item} tiles ({percent:.1f}%) procesados.")
                pre_ms, pre_pan, post_ms, post_pan = get_tile_from_bands(img_paths, y, x, tile_size)
                if pre_ms is not None and pre_pan is not None and post_ms is not None and post_pan is not None:
                    tile_pre = pansharpen_if_pan(pre_ms, pre_pan)
                    tile_post = pansharpen_if_pan(post_ms, post_pan)
                elif pre_ms is not None and pre_pan is not None:
                    tile_pre = pansharpen_if_pan(pre_ms, pre_pan)
                    tile_post = pre_ms.copy() # No pansharpen post
                elif post_ms is not None and post_pan is not None:
                    tile_pre = np.expand_dims(post_ms, 0) # No pansharpen pre
                    tile_post = post_ms.copy()
                else:
                    tqdm.write(f"[WARN] No MS ni PAN en escena {id_} de {city}, se omite tile {y},{x}.")
                    continue
                tile_post = coregister(tile_pre, tile_post)
                tile_mask = mask[y:y+tile_size, x:x+tile_size]
                if tile_mask.sum() == 0:
                    continue
                tile_img = np.concatenate([tile_pre, tile_post], axis=0)
                if stats is None:
                    stats = OnlineStats(tile_img.shape[0])
                    tqdm.write(f"[INFO] Inicializando OnlineStats para {city} con {tile_img.shape[0]} canales.")
                tile_id = f"{id_}_{y}_{x}"
                tile_img_t = torch.from_numpy(tile_img).float()
                tile_img_t = (tile_img_t - tile_img_t.mean()) / (tile_img_t.std() + 1e-6)
                stats.update(tile_img_t.numpy())
                tile_img_t = tile_img_t.half()
                tile_mask_t = torch.from_numpy(tile_mask).to(torch.uint8)
                unique_vals = np.unique(tile_mask)
                is_partial = is_partial_tile(tile_img)
                if (unique_vals == 255).all():
                    img_path = output_root / "test" / "images" / f"{tile_id}.pt"
                    mask_path = output_root / "test" / "masks" / f"{tile_id}.pt"
                    save_tensor(tile_img_t, img_path)
                    save_tensor(tile_mask_t, mask_path)
                    meta['test'].append({
                        "id": tile_id,
                        "city": city,
                        "scene_id": id_,
                        "bands": list(img_paths["pre"].keys()) + list(img_paths["post"].keys()),
                        "img_paths": {k: str(v) for k, v in img_paths.items()},
                        "label_path": str(label_path),
                        "bbox": [y, x, y+tile_size, x+tile_size],
                        "is_partial": is_partial,
                        "images_info": item["images_info"], # Añadir 'images_info' a cada entrada
                    })
                elif set(unique_vals).issubset({0, 1}):
                    img_path = output_root / "full" / "images" / f"{tile_id}.pt"
                    mask_path = output_root / "full" / "masks" / f"{tile_id}.pt"
                    if not img_path.exists():
                        save_tensor(tile_img_t, img_path)
                    if not mask_path.exists():
                        save_tensor(tile_mask_t, mask_path)
                    meta['full'].append({
                        "id": tile_id,
                        "city": city,
                        "scene_id": id_,
                        "bands": list(img_paths["pre"].keys()) + list(img_paths["post"].keys()),
                        "img_paths": {k: str(v) for k, v in img_paths.items()},
                        "label_path": str(label_path),
                        "bbox": [y, x, y+tile_size, x+tile_size],
                        "is_partial": is_partial,
                        "images_info": item["images_info"], # Añadir 'images_info' a cada entrada
                    })
                    if is_in_small_split(tile_id, small_split):
                        img_path = output_root / "small" / "images" / f"{tile_id}.pt"
                        mask_path = output_root / "small" / "masks" / f"{tile_id}.pt"
                        if not img_path.exists():
                            save_tensor(tile_img_t, img_path)
                        if not mask_path.exists():
                            save_tensor(tile_mask_t, mask_path)
                        meta['small'].append({
                            "id": tile_id,
                            "city": city,
                            "scene_id": id_,
                            "bands": list(img_paths["pre"].keys()) + list(img_paths["post"].keys()),
                            "img_paths": {k: str(v) for k, v in img_paths.items()},
                            "label_path": str(label_path),
                            "bbox": [y, x, y+tile_size, x+tile_size],
                            "is_partial": is_partial,
                            "images_info": item["images_info"], # Añadir 'images_info' a cada entrada
                        })
                else:
                    if count_mixed_tiles:
                        ignored_mixed_tiles += 1
                if debug_vis and len(debug_samples) < debug_vis:
                    debug_samples.append((tile_img, tile_mask, tile_id))
                del tile_img, tile_img_t, tile_mask, tile_mask_t, tile_pre, tile_post, pre_ms, pre_pan, post_ms, post_pan
                gc.collect()
        del mask
        gc.collect()
    tqdm.write(f"[INFO] Finalizado procesamiento de ciudad: {city}")
    if count_mixed_tiles:
        return (city, meta, stats, debug_samples), ignored_mixed_tiles
    else:
        return city, meta, stats, debug_samples

# --- main ---
def main():
    import argparse
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Preprocess earthquake-building-damage dataset.")
    parser.add_argument("--raw_data_root", type=str, default=str(RAW_DATA_ROOT), help="Raw data root directory")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT), help="Output root directory")
    parser.add_argument("--small_pct", type=float, default=0.1, help="Fraction of tiles per city for small split")
    parser.add_argument("--binary", action="store_true", help="Collapse mask to binary (0/1)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--debug_vis", type=int, default=0, help="Save N debug overlay PNGs")
    parser.add_argument("--prueba", action="store_true", help="Procesar solo una ciudad para pruebas rápidas",default=True)

    args = parser.parse_args()
    raw_data_root = Path(args.raw_data_root)
    output_root = Path(args.output_root)
    # Verificación de existencia de la ruta de datos
    if not raw_data_root.exists():
        raise FileNotFoundError(f"El directorio de datos no existe: {raw_data_root.resolve()}\n"
                                f"Por favor, revisa la ruta y pásala correctamente con --raw_data_root.")
    small_pct = args.small_pct
    binary = args.binary
    workers = args.workers
    debug_vis = args.debug_vis
    prueba = args.prueba
    tqdm.write(f"[INFO] Iniciando preprocesamiento con {workers} workers...")
    all_items = list(read_stac(raw_data_root))
    cities = sorted(set(item['city'] for item in all_items))
    tqdm.write(f"[INFO] Ciudades encontradas: {cities}")
    city_items = {c: [] for c in cities}
    for item in all_items:
        city_items[item['city']].append(item)
    # Ordenar ciudades por número de escenas (de menor a mayor)
    cities_sorted = sorted(cities, key=lambda c: len(city_items[c]))
    if prueba:
        # Solo procesar la primera ciudad para pruebas
        if cities_sorted:
            cities_sorted = [cities_sorted[0]]
            tqdm.write(f"[MODO PRUEBA] Solo se procesará la ciudad: {cities_sorted[0]}")
    for split in ["full", "small", "test"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "masks").mkdir(parents=True, exist_ok=True)
    params = {
        'val_split': False, 'small_split': small_pct, 'binary': binary, 'debug_vis': 0,
        'output_root': output_root, 'tile_size': TILE_SIZE, 'tile_overlap': TILE_OVERLAP
    }
    results = []
    ignored_mixed_tiles = 0  # Para validación de integridad
    if workers > 1:
        args_list = []
        for city in cities_sorted:
            params_city = params.copy()
            params_city['debug_vis'] = debug_vis
            args_list.append((city, city_items[city], params_city))
        with tqdm(total=len(cities_sorted), desc="Procesando ciudades", ncols=80) as pbar:
            with mp.Pool(workers) as pool:
                for res in pool.imap_unordered(process_city, args_list):
                    results.append(res[0])
                    ignored_mixed_tiles += res[1]
                    pbar.update(1)
    else:
        for city in tqdm(cities_sorted, desc="Procesando ciudades", ncols=80):
            params_city = params.copy()
            params_city['debug_vis'] = debug_vis
            res = process_city((city, city_items[city], params_city))
            results.append(res[0])
            ignored_mixed_tiles += res[1]
    meta = {s: [] for s in ['full', 'small', 'test']}
    stats = None
    debug_samples = []
    for city, meta_city, stats_city, debug_city in results:
        for split in meta:
            meta[split].extend(meta_city[split])
        if stats is None:
            stats = stats_city
            tqdm.write(f"[INFO] Inicializando OnlineStats global.")
        else:
            stats.merge(stats_city)
            tqdm.write(f"[INFO] Merge de OnlineStats para ciudad: {city}")
        debug_samples.extend(debug_city)
    for split in meta:
        tqdm.write(f"[INFO] Guardando meta.json para split: {split} ({len(meta[split])} tiles)")
        with open(output_root / split / "meta.json", "w") as f:
            json.dump(meta[split], f, indent=2)
    if stats is not None:
        mean, std = stats.finalize()
        tqdm.write(f"[INFO] Guardando stats.json (media y std globales)")
        with open(output_root / "stats.json", "w") as f:
            json.dump({"mean": mean, "std": std}, f, indent=2)
    if debug_vis and debug_samples:
        tqdm.write(f"[INFO] Guardando overlays de debug ({debug_vis}) en debug_vis/")
        random.shuffle(debug_samples)
        for tile_img, tile_mask, tile_id in debug_samples[:debug_vis]:
            out_path = output_root / "debug_vis" / f"{tile_id}.png"
            save_debug_vis(tile_img, tile_mask, out_path)
    # --- Validación de integridad ---
    tqdm.write("[VALIDACIÓN] Resumen de splits:")
    for split in meta:
        tqdm.write(f"  Split {split}: {len(meta[split])} tiles")
    tqdm.write(f"  Tiles mixtos ignorados: {ignored_mixed_tiles}")
    tqdm.write(f"[INFO] Preprocesamiento finalizado.")
    end_time = time.time()
    elapsed = end_time - start_time
    tqdm.write(f"[INFO] Tiempo total de ejecución: {elapsed:.2f} segundos.")

# --- Nueva función para detectar si un tile es parcial (tiene valores nodata o NaN)
def is_partial_tile(tile_img, nodata_values=[0, -9999]):
    # tile_img: (C, H, W)
    # Considera parcial si hay algún NaN o algún valor nodata en cualquier banda
    if np.isnan(tile_img).any():
        return True
    for nodata in nodata_values:
        if (tile_img == nodata).any():
            return True
    return False

if __name__ == "__main__":
    main() 