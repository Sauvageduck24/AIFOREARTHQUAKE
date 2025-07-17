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

RAW_DATA_ROOT = Path("./data/international-charter-eo4ai-etq-challenge/Training")
OUTPUT_ROOT = Path("./data/preprocessed")
TILE_SIZE = 512
TILE_OVERLAP = 64

# --- read_stac adaptado a la estructura real ---
def read_stac(raw_data_root: Path, annotations_root: Path = None):
    annotations_root = annotations_root or (raw_data_root / "Annotations")
    tqdm.write(f"[INFO] Buscando ciudades en {raw_data_root} ...")
    for city_dir in raw_data_root.iterdir():
        if not city_dir.is_dir():
            continue
        collection_path = city_dir / "collection.json"
        if not collection_path.exists():
            tqdm.write(f"[WARN] No collection.json en {city_dir}, se omite.")
            continue
        tqdm.write(f"[INFO] Procesando ciudad: {city_dir.name}")
        with open(collection_path) as f:
            collection = json.load(f)
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
            # Iterar sobre escenas
            for link in collection.get("links", []):
                if link.get("rel") != "item":
                    continue
                scene_json = city_dir / link["href"]
                if not scene_json.exists():
                    tqdm.write(f"[WARN] No existe escena {scene_json}, se omite.")
                    continue
                with open(scene_json) as sf:
                    scene = json.load(sf)
                assets = scene.get("assets", {})
                img_paths = {k: (scene_json.parent / v["href"]).resolve() for k, v in assets.items() if "href" in v}
                yield {
                    "city": city_dir.name,
                    "id": f"{scene.get('id', scene_json.stem)}__{label_path.stem}",
                    "img_paths": img_paths,  # dict: band -> path
                    "label_path": str(label_path),
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
    dy, dx, _ = phase_cross_correlation(pre_mean, post_mean, upsample_factor=10)
    dy = float(np.atleast_1d(dy)[0])
    dx = float(np.atleast_1d(dx)[0])
    return ndi_shift(post, shift=(0, dy, dx), mode="nearest")

# --- clip_to_aoi fix ---
def clip_to_aoi(img: np.ndarray, gpkg_path: Path, raster_profile: dict):
    gdf = gpd.read_file(gpkg_path)
    bounds = gdf.total_bounds
    transform = raster_profile["transform"]
    row_start, col_start = ~transform * (bounds[0], bounds[3])
    row_stop,  col_stop  = ~transform * (bounds[2], bounds[1])
    row_min, row_max = sorted([int(row_start), int(row_stop)])
    col_min, col_max = sorted([int(col_start), int(col_stop)])
    clipped = img[:, row_min:row_max, col_min:col_max]
    new_profile = raster_profile.copy()
    new_profile["height"] = clipped.shape[1]
    new_profile["width"] = clipped.shape[2]
    new_profile["transform"] = rasterio.transform.from_origin(
        bounds[0], bounds[3], transform.a, transform.e
    )
    return clipped, new_profile

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
    if "damage_grade" in gdf.columns:
        values = [int(getattr(row, "damage_grade", 1)) for row in gdf.itertuples()]
    else:
        values = [1 for _ in gdf.geometry]
    burned = rasterio.features.rasterize(
        [(geom, v) for geom, v in zip(gdf.geometry, values)],
        out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8
    )
    if binary:
        burned = (burned > 0).astype(np.uint8)
    return burned

# --- tile_and_filter ---
def tile_and_filter(img, mask, tile_size, overlap):
    H, W = mask.shape
    step = tile_size - overlap
    for y in range(0, H - tile_size + 1, step):
        for x in range(0, W - tile_size + 1, step):
            tile_img = img[:, y:y+tile_size, x:x+tile_size]
            tile_mask = mask[y:y+tile_size, x:x+tile_size]
            if tile_mask.sum() == 0:
                continue
            yield tile_img, tile_mask, (y, x)

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
    ms_bands = [b for b in ["blue", "green", "red", "nir"] if b in img_paths]
    pan_band = "pan" if "pan" in img_paths else None
    tile_ms = None
    tile_pan = None
    if ms_bands:
        tile_ms = []
        for b in ms_bands:
            with rasterio.open(str(img_paths[b])) as src:
                tile = src.read(1, window=rasterio.windows.Window(x, y, tile_size, tile_size)).astype(np.float32)
                tile_ms.append(tile)
        tile_ms = np.stack(tile_ms, axis=0)
    if pan_band:
        with rasterio.open(str(img_paths[pan_band])) as src:
            tile_pan = src.read(1, window=rasterio.windows.Window(x, y, tile_size, tile_size)).astype(np.float32)
    return tile_ms, tile_pan

# --- per-city processing ---
def process_city(args):
    (city, items, params) = args
    tqdm.write(f"[INFO] Iniciando procesamiento de ciudad: {city} ({len(items)} escenas)")
    val_split = params['val_split']
    small_split = params['small_split']
    binary = params['binary']
    debug_vis = params['debug_vis']
    output_root = params['output_root']
    tile_size = params['tile_size']
    tile_overlap = params['tile_overlap']
    stats = None
    meta = {s: [] for s in ['full', 'small', 'val']}
    debug_samples = []
    import gc
    total_tiles = 0
    for item in items:
        id_ = item["id"]
        img_paths = item["img_paths"]
        label_path = item["label_path"]
        # Abrir una banda para obtener shape y transform
        with rasterio.open(str(list(img_paths.values())[0])) as src:
            H, W = src.height, src.width
            pre_profile = src.profile
            transform = src.transform
        # Rasterizar máscara completa una vez (puede ser optimizado a tile si es muy grande)
        mask = rasterize_labels(label_path, (H, W), transform, binary=binary)
        # Tileado robusto
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
                tile_ms, tile_pan = get_tile_from_bands(img_paths, y, x, tile_size)
                if tile_ms is not None and tile_pan is not None:
                    tile_pre = pansharpen_if_pan(tile_ms, tile_pan)
                    tile_post = tile_pre.copy()
                elif tile_ms is not None:
                    tile_pre = tile_ms
                    tile_post = tile_ms.copy()
                elif tile_pan is not None:
                    tile_pre = np.expand_dims(tile_pan, 0)
                    tile_post = tile_pre.copy()
                else:
                    tqdm.write(f"[WARN] No MS ni PAN en escena {id_} de {city}, se omite tile {y},{x}.")
                    continue
                # Coregister (no tiene sentido si pre==post, pero se deja por compatibilidad)
                tile_post = coregister(tile_pre, tile_post)
                # Clip to AOI (ya está en tile, así que solo shape)
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
                tile_img_t = tile_img_t.half()
                tile_mask_t = torch.from_numpy(tile_mask).to(torch.uint8)
                stats.update(tile_img_t.float().numpy())
                for split in ['full', 'small', 'val']:
                    if split == 'small' and not is_in_small_split(tile_id, small_split):
                        continue
                    if split == 'val' and not val_split:
                        continue
                    img_path = output_root / "full" / "images" / f"{tile_id}.pt"
                    mask_path = output_root / "full" / "masks" / f"{tile_id}.pt"
                    if not img_path.exists():
                        save_tensor(tile_img_t, img_path)
                    if not mask_path.exists():
                        save_tensor(tile_mask_t, mask_path)
                    meta[split].append({
                        "id": tile_id,
                        "city": city,
                        "scene_id": id_,
                        "bands": list(img_paths.keys()),
                        "img_paths": {k: str(v) for k, v in img_paths.items()},
                        "label_path": str(label_path),
                        "bbox": [y, x, y+tile_size, x+tile_size],
                    })
                if debug_vis and len(debug_samples) < debug_vis:
                    debug_samples.append((tile_img, tile_mask, tile_id))
                del tile_img, tile_img_t, tile_mask, tile_mask_t, tile_pre, tile_post, tile_ms, tile_pan
                gc.collect()
        del mask
        gc.collect()
    tqdm.write(f"[INFO] Finalizado procesamiento de ciudad: {city}")
    return city, meta, stats, debug_samples

# --- main ---
def main():
    import argparse
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Preprocess earthquake-building-damage dataset.")
    parser.add_argument("--raw_data_root", type=str, default=str(RAW_DATA_ROOT), help="Raw data root directory")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT), help="Output root directory")
    parser.add_argument("--val_pct", type=float, default=0.1, help="Fraction of cities for val split")
    parser.add_argument("--small_pct", type=float, default=0.1, help="Fraction of tiles per city for small split")
    parser.add_argument("--binary", action="store_true", help="Collapse mask to binary (0/1)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--debug_vis", type=int, default=0, help="Save N debug overlay PNGs")

    args = parser.parse_args()
    raw_data_root = Path(args.raw_data_root)
    output_root = Path(args.output_root)
    val_pct = args.val_pct
    small_pct = args.small_pct
    binary = args.binary
    workers = args.workers
    debug_vis = args.debug_vis
    tqdm.write(f"[INFO] Iniciando preprocesamiento con {workers} workers...")
    all_items = list(read_stac(raw_data_root))
    cities = sorted(set(item['city'] for item in all_items))
    tqdm.write(f"[INFO] Ciudades encontradas: {cities}")
    val_cities = {c for c in cities if is_val_city(c, val_pct)}
    city_items = {c: [] for c in cities}
    for item in all_items:
        city_items[item['city']].append(item)
    # Ordenar ciudades por número de escenas (de menor a mayor)
    cities_sorted = sorted(cities, key=lambda c: len(city_items[c]))
    for split in ["full", "small", "val"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "masks").mkdir(parents=True, exist_ok=True)
    params = {
        'val_split': False, 'small_split': small_pct, 'binary': binary, 'debug_vis': 0,
        'output_root': output_root, 'tile_size': TILE_SIZE, 'tile_overlap': TILE_OVERLAP
    }
    results = []
    if workers > 1:
        args_list = []
        for city in cities_sorted:
            params_city = params.copy()
            params_city['val_split'] = city in val_cities
            params_city['debug_vis'] = debug_vis
            args_list.append((city, city_items[city], params_city))
        with mp.Pool(workers) as pool:
            results = pool.map(process_city, args_list)
    else:
        for city in cities_sorted:
            params_city = params.copy()
            params_city['val_split'] = city in val_cities
            params_city['debug_vis'] = debug_vis
            results.append(process_city((city, city_items[city], params_city)))
    meta = {s: [] for s in ['full', 'small', 'val']}
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
    tqdm.write(f"[INFO] Preprocesamiento finalizado.")
    end_time = time.time()
    elapsed = end_time - start_time
    tqdm.write(f"[INFO] Tiempo total de ejecución: {elapsed:.2f} segundos.")

if __name__ == "__main__":
    main() 