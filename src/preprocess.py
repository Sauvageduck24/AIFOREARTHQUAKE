"""
Earthquake Building Damage Preprocessing Pipeline
------------------------------------------------
Robust pipeline for preparing the "international-charter-eo4ai-etq-challenge" dataset.

Major Improvements:
1. Efficient raster reading with windowed processing
2. Enhanced coregistration with fallback mechanisms
3. Proper STAC metadata handling
4. Optimized memory management
5. Comprehensive error handling
6. Configurable tile filtering
"""

import os
import json
import hashlib
from pathlib import Path
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from skimage.transform import warp
import rasterio.features
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy.ndimage import shift as ndi_shift
import re
import argparse
import time
from typing import Dict, List, Tuple, Optional, Generator
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
import itertools
import shutil
# Añadir rich para mejorar la salida de consola
from rich.console import Console
from rich.text import Text
from rich.progress import track

console = Console()

# Constants
RAW_DATA_ROOT = Path("data/international-charter-eo4ai-etq-challenge/Training")
OUTPUT_ROOT = Path("data/preprocessed")
TILE_SIZE = 512
TILE_OVERLAP = 64
VALID_BANDS = ['blue', 'green', 'red', 'nir', 'pan']
MIN_VALID_PIXELS = 0.01  # Porcentaje de exigencia para descartar tiles, cuanto mas alto mayor descarte
DEFAULT_SEED = 42

# 1. Definir estructura de estadísticas por banda y pre/post
BAND_NAMES = ['blue', 'green', 'red', 'nir']

class BandStats:
    def __init__(self):
        # stats['pre']['blue'], stats['post']['nir'], ...
        self.stats = {'pre': {b: OnlineStats(1) for b in BAND_NAMES},
                      'post': {b: OnlineStats(1) for b in BAND_NAMES}}
    def update(self, pre_ms, post_ms, bands):
        # pre_ms, post_ms: (n_bands, H, W), bands: list of band names
        for i, b in enumerate(bands):
            if b in self.stats['pre']:
                self.stats['pre'][b].update(pre_ms[i:i+1])
            if b in self.stats['post']:
                self.stats['post'][b].update(post_ms[i:i+1])
    def finalize(self):
        result = {'pre': {}, 'post': {}}
        for t in ['pre', 'post']:
            for b in BAND_NAMES:
                mean, std = self.stats[t][b].finalize()
                result[t][b] = {'mean': mean[0] if mean else None, 'std': std[0] if std else None}
        return result

class OnlineStats:
    def __init__(self, n_channels):
        self.count = 0
        self.sum   = np.zeros(n_channels, np.float64)
        self.sumsq = np.zeros(n_channels, np.float64)

    def update(self, x):              # x (C,H,W)
        x = x.reshape(x.shape[0], -1)
        self.count += x.shape[1]
        self.sum   += x.sum(1)
        self.sumsq += (x ** 2).sum(1)

    def merge(self, other):
        self.count += other.count
        self.sum   += other.sum
        self.sumsq += other.sumsq

    def finalize(self):
        if self.count == 0:
            return [], []
        mean = self.sum / self.count
        var  = self.sumsq / self.count - mean**2
        std  = np.sqrt(np.maximum(var, 0))
        return mean.tolist(), std.tolist()


def read_stac(raw_data_root: Path) -> Generator[Dict, None, None]:
    """
    Robust STAC reader with comprehensive error handling
    Returns generator of scene dictionaries
    """
    annotations_root = raw_data_root / "Annotations"
    city_dirs = [d for d in raw_data_root.iterdir() if d.is_dir() and d.name.startswith("Call_")]
    
    if not city_dirs:
        console.print(f"[bold red][ERROR][/bold red] No valid city directories found in {raw_data_root}")
        raise FileNotFoundError(f"No valid city directories found in {raw_data_root}")
    
    for city_dir in track(city_dirs, description="[cyan]Processing cities"):  # reemplaza tqdm
        collection_path = city_dir / "collection.json"
        if not collection_path.exists():
            console.print(f"[yellow][WARN][/yellow] Skipping {city_dir.name}: collection.json missing")
            continue
            
        try:
            with open(collection_path) as f:
                collection = json.load(f)
        except json.JSONDecodeError:
            console.print(f"[red][ERROR][/red] Invalid JSON in {collection_path}")
            continue
            
        # Extract event date
        event_date = None
        desc = collection.get("description", "")
        date_match = re.search(r"Date of Charter Activation[:\s]+(\d{4}-\d{2}-\d{2})", desc)
        if date_match:
            event_date = f"{date_match.group(1)}T00:00:00Z"
        else:
            console.print(f"[yellow][WARN][/yellow] {city_dir.name}: Event date not found")
            continue
            
        # Find annotation files
        ann_city_dir = annotations_root / city_dir.name
        if not ann_city_dir.exists():
            console.print(f"[yellow][WARN][/yellow] Skipping {city_dir.name}: Annotation directory missing")
            continue
            
        gpkg_files = list(ann_city_dir.glob("*.gpkg"))
        if not gpkg_files:
            console.print(f"[yellow][WARN][/yellow] Skipping {city_dir.name}: No annotation files")
            continue
            
        # Process scenes
        pre_items, post_items = [], []
        for link in collection.get("links", []):
            if link.get("rel") != "item":
                continue
                
            scene_path = city_dir / link["href"]
            if not scene_path.exists():
                console.print(f"[yellow][WARN][/yellow] Scene not found: {scene_path}")
                continue
                
            try:
                with open(scene_path) as sf:
                    scene = json.load(sf)
            except json.JSONDecodeError:
                console.print(f"[red][ERROR][/red] Invalid scene JSON: {scene_path}")
                continue
                
            # Extract scene date
            scene_date = scene.get("properties", {}).get("datetime")
            if not scene_date:
                date_match = re.search(r"(\d{8})", scene_path.name)
                if date_match:
                    d = date_match.group(1)
                    scene_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}T00:00:00Z"
            
            if not scene_date:
                console.print(f"[yellow][WARN][/yellow] Skipping scene: Date not found in {scene_path.name}")
                continue
                
            if scene_date < event_date:
                pre_items.append(scene)
            else:
                post_items.append(scene)
        
        if not pre_items or not post_items:
            console.print(f"[yellow][WARN][/yellow] Skipping {city_dir.name}: Missing pre/post scenes")
            continue
            
        # Process annotations
        for label_path in gpkg_files:
            # Find best pre/post pair based on temporal proximity
            pre_scene = min(pre_items, key=lambda x: abs(np.datetime64(x['properties']['datetime'].replace('Z', '')) - np.datetime64(event_date.replace('Z', ''))))
            post_scene = min(post_items, key=lambda x: abs(np.datetime64(x['properties']['datetime'].replace('Z', '')) - np.datetime64(event_date.replace('Z', ''))))
            
            # Extract image paths
            img_paths = {
                "pre": _extract_asset_paths(pre_scene, city_dir),
                "post": _extract_asset_paths(post_scene, city_dir)
            }
            
            # Validate we have required bands
            if not _validate_bands(img_paths["pre"]) or not _validate_bands(img_paths["post"]):
                console.print(f"[yellow][WARN][/yellow] Skipping {city_dir.name}: Missing required bands")
                continue
                
            yield {
                "city": city_dir.name,
                "id": f"{pre_scene['id']}_{post_scene['id']}__{label_path.stem}",
                "img_paths": img_paths,
                "label_path": str(label_path),
                "pre_date": pre_scene['properties']['datetime'],
                "post_date": post_scene['properties']['datetime'],
                "event_date": event_date
            }

def _extract_asset_paths(scene: Dict, base_dir: Path) -> Dict[str, Path]:
    """Resolve asset paths with robust error handling"""
    paths = {}
    for band, asset in scene.get("assets", {}).items():
        if band not in VALID_BANDS:
            continue
            
        rel_path = asset.get("href")
        if not rel_path:
            continue
            
        abs_path = (base_dir / rel_path).resolve()
        if abs_path.exists():
            paths[band] = abs_path
        else:
            # Search recursively
            found = list(base_dir.glob(f"**/{rel_path}"))
            if found:
                paths[band] = found[0].resolve()
            else:
                console.print(f"[yellow][WARN][/yellow] Asset not found: {rel_path}")
    return paths

def _validate_bands(band_dict: Dict[str, Path]) -> bool:
    """Check if we have minimum required bands (RGB)"""
    return all(b in band_dict for b in ['red', 'green', 'blue'])

def pansharpen(ms_img: np.ndarray, pan_img: np.ndarray) -> np.ndarray:
    """
    High-quality pansharpening using Gram-Schmidt Adaptive (GSA) method
    Based on: https://gdal.org/programs/gdal_pansharpen.html
    """
    # Step 1: Create low-resolution pan by averaging MS bands
    lr_pan = ms_img.mean(axis=0)
    
    # Step 2: Compute gain coefficients
    gains = []
    for band in ms_img:
        cov = np.cov(band.flatten(), pan_img.flatten())[0, 1]
        var = np.var(lr_pan.flatten())
        gains.append(cov / var if var > 0 else 1.0)
    
    # Step 3: Apply transformation
    pan_sharpened = np.zeros_like(ms_img)
    for i, (band, gain) in enumerate(zip(ms_img, gains)):
        # Compute difference between HR pan and LR pan
        detail = pan_img - lr_pan
        # Apply detail to MS band
        pan_sharpened[i] = band + gain * detail
    
    return pan_sharpened

def coregister(
    base_img: np.ndarray, 
    moving_img: np.ndarray,
    max_shift: int = 100
) -> np.ndarray:
    """
    Robust coregistration with fallback to original on failure
    Returns registered moving image
    """
    # Handle edge cases
    if base_img.shape != moving_img.shape:
        raise ValueError("Base and moving images must have same shape")
    # Use mean of bands for grayscale registration
    base_gray = base_img.mean(axis=0) if base_img.ndim == 3 else base_img
    moving_gray = moving_img.mean(axis=0) if moving_img.ndim == 3 else moving_img
    # Check for empty images
    if np.all(base_gray == 0) or np.all(moving_gray == 0):
        return moving_img
    try:
        # Compute shift (compatible with older skimage)
        shift_result = phase_cross_correlation(
            base_gray, 
            moving_gray, 
            upsample_factor=10
        )
        # shift_result puede ser (shift, error, diffphase) o solo (shift, error)
        shift = shift_result[0] if isinstance(shift_result, tuple) else shift_result
        # Apply shift
        registered = ndi_shift(
            moving_img, 
            shift=(0, shift[0], shift[1]), 
            mode='nearest'
        )
        return registered
    except Exception as e:
        console.print(f"[yellow][WARN][/yellow] Coregistration failed: {str(e)}")
        return moving_img

def rasterize_labels(
    gpkg_path: Path, 
    transform: rasterio.Affine,
    shape: Tuple[int, int],
    crs_raster,  # nuevo argumento
    binary: bool = False
) -> np.ndarray:
    """
    Efficient label rasterization with error handling
    """
    try:
        gdf = gpd.read_file(gpkg_path)
        if VERBOSE:
            print(f"[RASTERIZE] GPKG CRS: {gdf.crs}")
            print(f"[RASTERIZE] GPKG bounds: {gdf.total_bounds}")
            print(f"[RASTERIZE] GPKG rows: {len(gdf)}")
        if VERBOSE:
            print(f"[RASTERIZE] Raster shape: {shape}")
            print(f"[RASTERIZE] Raster transform: {transform}")
        # Reproyectar si es necesario
        if gdf.crs != crs_raster:
            gdf = gdf.to_crs(crs_raster)
            if VERBOSE:
                print(f"[RASTERIZE] GPKG reproyectado a: {gdf.crs}")
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
        
        if gdf.empty:
            return np.zeros(shape, dtype=np.uint8)
        
        # Handle damage classification (ajustado para binarizar y marcar desconocidos)
        if "damaged" in gdf.columns:
            values = []
            for val in gdf["damaged"]:
                if pd.isna(val) or val is None:
                    values.append(255)  # Unknown
                elif int(val) == 1:
                    values.append(1)
                elif int(val) == 0:
                    values.append(0)
                else:
                    values.append(255)  # Cualquier otro valor inesperado
        else:
            values = [1] * len(gdf)
        
        mask = rasterio.features.rasterize(
            zip(gdf.geometry, values),
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        return mask
    except Exception as e:
        console.print(f"[red][ERROR][/red] Rasterization failed for {gpkg_path}: {str(e)}")
        return np.zeros(shape, dtype=np.uint8)

def process_scene(scene: Dict, output_root: Path, tile_size: int, tile_overlap: int,
                  binary: bool, debug_vis: int) -> Tuple[Dict, BandStats, List]:
    with rasterio.open(scene["img_paths"]["pre"]["red"]) as ref:
        transform, (height, width) = ref.transform, (ref.height, ref.width)
        crs_raster = ref.crs
        if VERBOSE:
            print(f"[PROCESS_SCENE] Raster CRS: {ref.crs}")
            print(f"[PROCESS_SCENE] Raster bounds: {ref.bounds}")
            print(f"[PROCESS_SCENE] Raster shape: {(height, width)}")
            print(f"[PROCESS_SCENE] Raster transform: {transform}")

    mask = rasterize_labels(Path(scene["label_path"]), transform, (height, width), crs_raster, binary)
    if VERBOSE:
        print(f"Valores únicos en la máscara de la escena {scene['id']}: {np.unique(mask)}")

    y_starts = range(0, height, tile_size - tile_overlap)
    x_starts = range(0, width,  tile_size - tile_overlap)

    pre_ds  = {b: rasterio.open(p) for b, p in scene["img_paths"]["pre"].items()}
    post_ds = {b: rasterio.open(p) for b, p in scene["img_paths"]["post"].items()}
    band_stats = BandStats()
    meta    = {s: [] for s in ['full', 'small', 'test']}
    debug   = []

    try:
        total_tiles = len(y_starts) * len(x_starts)
        for y, x in tqdm(list(itertools.product(y_starts, x_starts)), desc=f"Tiles {scene['id']}",total=total_tiles, leave=False):
            win = Window(x, y, tile_size, tile_size)
            tile_mask = mask[y:y+tile_size, x:x+tile_size]
            valid_pct = np.mean(tile_mask != 0)
            if np.all(tile_mask == 0) or valid_pct < MIN_VALID_PIXELS:
                continue

            # ── lectura rápida ─────────────────────────────────────────────
            ms_bands = ['blue', 'green', 'red', 'nir']
            common_bands = [b for b in ms_bands if b in pre_ds and b in post_ds]
            if not common_bands:
                if VERBOSE:
                    print(f"[WARN] Tile ({y},{x}) sin bandas multiespectrales en común. Se omite.")
                continue

            pre_ms  = np.stack([pre_ds[b].read(1, window=win, boundless=True) for b in common_bands], axis=0).astype(np.float32)
            post_ms = np.stack([post_ds[b].read(1, window=win, boundless=True) for b in common_bands], axis=0).astype(np.float32)

            if 'pan' in pre_ds:
                pre_ms  = pansharpen(pre_ms,  pre_ds['pan'].read(1, window=win, boundless=True))
            if 'pan' in post_ds:
                post_ms = pansharpen(post_ms, post_ds['pan'].read(1, window=win, boundless=True))

            if pre_ms.shape == post_ms.shape:
                post_ms = coregister(pre_ms, post_ms)
            else:
                if VERBOSE:
                    print(f"[WARN] pre_ms y post_ms tienen shapes diferentes: {pre_ms.shape} vs {post_ms.shape}. Se omite coregister.")

            # Actualizar estadísticas por banda
            band_stats.update(pre_ms, post_ms, common_bands)
            tile_meta = _process_tile(scene, pre_ms, post_ms, tile_mask,
                                      (y, x), output_root, binary, band_stats, common_bands)
            if tile_meta:
                meta[tile_meta['split']].append(tile_meta)
                if debug_vis and len(debug) < debug_vis:
                    debug.append((pre_ms[[2,1,0]] if pre_ms.shape[0] >= 3 else pre_ms[0:3], tile_mask, tile_meta['id']))
    finally:
        for ds in itertools.chain(pre_ds.values(), post_ds.values()):
            ds.close()

    return meta, band_stats, debug

# 3. En _process_tile, quitar stats.update y solo dejar la lógica de guardado y meta
# (band_stats se pasa pero no se usa aquí, solo para mantener la firma)
def _process_tile(
    scene: Dict,
    pre_ms: np.ndarray,
    post_ms: np.ndarray,
    tile_mask: np.ndarray,
    position: Tuple[int, int],
    output_root: Path,
    binary: bool,
    band_stats: BandStats,
    common_bands: list
) -> Optional[Dict]:
    y, x = position
    valid_pct = np.mean(tile_mask != 0)
    #print(f"Tile ({y},{x}) - Porcentaje de píxeles válidos: {valid_pct:.4f}, valores únicos: {np.unique(tile_mask)}")
    if np.all(tile_mask == 0) or valid_pct < MIN_VALID_PIXELS:
        return None
    tile_img = np.concatenate([pre_ms, post_ms], axis=0)
    tile_id = f"{scene['id']}_{y}_{x}"
    unique_mask = np.unique(tile_mask)
    if any((v not in (0, 1)) for v in unique_mask):
        split = 'test'
    else:
        split = 'full'
    # Guardar como .npz comprimido en vez de torch.save
    img_path = output_root / split / "images" / f"{tile_id}.npz"
    mask_path = output_root / split / "masks" / f"{tile_id}.npz"
    np.savez_compressed(img_path, tile_img.astype(np.float16))
    np.savez_compressed(mask_path, tile_mask.astype(np.uint8))
    # Construir images_info
    images_info = {
        'pre': [
            {
                'band': b,
                'path': str(scene['img_paths']['pre'][b]),
                'date': scene['pre_date']
            } for b in common_bands if b in scene['img_paths']['pre']
        ],
        'post': [
            {
                'band': b,
                'path': str(scene['img_paths']['post'][b]),
                'date': scene['post_date']
            } for b in common_bands if b in scene['img_paths']['post']
        ]
    }
    # Determinar si el tile es parcial (fuera de la imagen o contiene datos nulos)
    is_partial = False
    if (
        np.isnan(pre_ms).any() or np.isnan(post_ms).any() or
        pre_ms.shape[1:] != (TILE_SIZE, TILE_SIZE) or post_ms.shape[1:] != (TILE_SIZE, TILE_SIZE)
    ):
        is_partial = True

    return {
        "id": tile_id,
        "city": scene["city"],
        "split": split,
        "position": [y, x],
        "size": TILE_SIZE,
        "pre_date": scene["pre_date"],
        "post_date": scene["post_date"],
        "event_date": scene["event_date"],
        "img_path": str(img_path),
        "mask_path": str(mask_path),
        "has_damage": bool((tile_mask == 1).any()),
        "bands": common_bands,
        "images_info": images_info,
        "is_partial": is_partial
    }



def save_debug_vis(tile_img: np.ndarray, tile_mask: np.ndarray, out_path: Path):
    """Save debug visualization"""
    plt.figure(figsize=(10, 5))
    
    # Pre-event image
    plt.subplot(121)
    rgb = np.stack([
        tile_img[2],  # Red
        tile_img[1],  # Green
        tile_img[0]   # Blue
    ], axis=-1)
    plt.imshow(np.clip(rgb / np.percentile(rgb, 99), 0, 1))
    plt.title("Pre-event")
    
    # Damage overlay
    plt.subplot(122)
    plt.imshow(np.clip(rgb / np.percentile(rgb, 99), 0, 1))
    plt.imshow(tile_mask, alpha=0.5, cmap='jet', vmin=0, vmax=2)
    plt.title("Damage overlay")
    
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    print()
    parser = argparse.ArgumentParser(description="Preprocess earthquake-building-damage dataset.")
    parser.add_argument("--raw_data_root", type=str, default=str(RAW_DATA_ROOT), help="Raw data root directory")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT), help="Output root directory")
    parser.add_argument("--small_pct", type=float, default=0.1, help="Fraction of tiles per city for small split")
    parser.add_argument("--binary", action="store_true", help="Collapse mask to binary (0/1)",default=True)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--debug_vis", type=int, default=0, help="Save N debug overlay PNGs")
    parser.add_argument("--verbose", action="store_true", help="Mostrar información detallada de depuración",default=False)
    parser.add_argument("--prueba", action="store_true", help="Procesar solo una ciudad para pruebas rápidas",default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed para reproducibilidad")
    args = parser.parse_args()
    
    global VERBOSE
    VERBOSE = args.verbose

    # Fijar seed para reproducibilidad
    import random as pyrandom
    pyrandom.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Borrar carpeta de salida si existe
    output_root = Path(args.output_root)
    if output_root.exists():
        console.print(f"[bold yellow][INFO][/bold yellow] Borrando carpeta de salida: {output_root}")
        shutil.rmtree(output_root)

    # Validate paths
    raw_root = Path(args.raw_data_root)
    if not raw_root.exists():
        console.print(f"[bold red][ERROR][/bold red] Raw data directory not found: {raw_root}")
        raise FileNotFoundError(f"Raw data directory not found: {raw_root}")
        
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create output structure
    for split in ['full', 'small', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Process scenes
    scenes = list(read_stac(raw_root))
    if not scenes:
        console.print(f"[bold red][ERROR][/bold red] No valid scenes found for processing")
        raise RuntimeError("No valid scenes found for processing")

    if args.prueba:
        console.print("[bold cyan][INFO][/bold cyan] Modo prueba activado: solo se procesarán 3 escenas.")
        scenes = scenes[:3]

    console.print(f"[bold green]Found {len(scenes)} scenes to process[/bold green]")
    
    # Process with multiprocessing
    params = {
        'output_root': output_root,
        'tile_size': TILE_SIZE,
        'tile_overlap': TILE_OVERLAP,
        'binary': args.binary,
        'debug_vis': args.debug_vis
    }
    
    all_meta = {s: [] for s in ['full', 'small', 'test']}
    all_band_stats = BandStats()
    debug_samples = []

    if args.workers == 1:
        total_time = 0
        for i, scene in enumerate(tqdm(scenes, desc="Processing scenes")):
            start_time = time.time()
            if VERBOSE:
                print(f"[INFO] Processing scene {i+1}/{len(scenes)}: {scene['id']}")
            meta, band_stats, debug = process_scene(
                scene,
                params['output_root'],
                params['tile_size'],
                params['tile_overlap'],
                params['binary'],
                params['debug_vis']
            )
            elapsed = time.time() - start_time
            total_time += elapsed
            tqdm.write(f"[INFO] Scene {i+1}/{len(scenes)} procesada en {elapsed:.2f} segundos. Tiempo acumulado: {total_time:.2f} segundos.")
            for split in all_meta:
                all_meta[split].extend(meta[split])
            all_band_stats.stats['pre'] = {b: all_band_stats.stats['pre'][b] for b in BAND_NAMES}
            all_band_stats.stats['post'] = {b: all_band_stats.stats['post'][b] for b in BAND_NAMES}
            for t in ['pre', 'post']:
                for b in BAND_NAMES:
                    all_band_stats.stats[t][b].merge(band_stats.stats[t][b])
            debug_samples.extend(debug)
    else:
        with mp.Pool(args.workers) as pool:
            results = []
            for scene in scenes:
                results.append(pool.apply_async(
                    process_scene, 
                    (
                        scene, 
                        params['output_root'], 
                        params['tile_size'], 
                        params['tile_overlap'], 
                        params['binary'], 
                        params['debug_vis']
                    )
                ))
            total_time = 0
            for i, res in enumerate(tqdm(results, desc="Processing scenes")):
                start_time = time.time()
                meta, band_stats, debug = res.get()
                elapsed = time.time() - start_time
                total_time += elapsed
                tqdm.write(f"[INFO] Scene {i+1}/{len(results)} procesada en {elapsed:.2f} segundos. Tiempo acumulado: {total_time:.2f} segundos.")
                for split in all_meta:
                    all_meta[split].extend(meta[split])
                all_band_stats.stats['pre'] = {b: all_band_stats.stats['pre'][b] for b in BAND_NAMES}
                all_band_stats.stats['post'] = {b: all_band_stats.stats['post'][b] for b in BAND_NAMES}
                for t in ['pre', 'post']:
                    for b in BAND_NAMES:
                        all_band_stats.stats[t][b].merge(band_stats.stats[t][b])
                debug_samples.extend(debug)
    
    # Hacer el split 'small' como 10% aleatorio de 'full' por ciudad
    pyrandom.seed(args.seed)
    full_tiles_by_city = {}
    for tile in all_meta['full']:
        city = tile['city']
        full_tiles_by_city.setdefault(city, []).append(tile)
    small_tiles = []
    new_full_tiles = []
    for city, tiles in full_tiles_by_city.items():
        n_small = max(1, int(0.1 * len(tiles)))
        idxs = list(range(len(tiles)))
        pyrandom.shuffle(idxs)
        small_idx = set(idxs[:n_small])
        for i, tile in enumerate(tiles):
            if i in small_idx:
                tile_copy = tile.copy()
                tile_copy['split'] = 'small'
                small_tiles.append(tile_copy)
            else:
                new_full_tiles.append(tile)
    all_meta['full'] = new_full_tiles
    all_meta['small'] = small_tiles

    # Save metadata
    for split in all_meta:
        with open(output_root / split / "meta.json", 'w') as f:
            json.dump(all_meta[split], f, indent=2)
    
    # Save statistics
    stats_result = all_band_stats.finalize()
    with open(output_root / "stats.json", 'w') as f:
        json.dump(stats_result, f, indent=2)
    
    # Save debug visuals
    if debug_samples:
        debug_dir = output_root / "debug_vis"
        debug_dir.mkdir(exist_ok=True)
        for i, (img, mask, tile_id) in enumerate(debug_samples[:args.debug_vis]):
            save_debug_vis(img, mask, debug_dir / f"{tile_id}.png")
    
    tqdm.write("Preprocessing completed successfully\n")

if __name__ == "__main__":
    main()