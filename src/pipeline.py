import os
import sys
import glob
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from tqdm import tqdm
from rasterio.transform import from_bounds
import rasterio
from preprocess_lib import (
    load_gpkg_with_fiona, find_scenes_for_aoi, buffer_geom, process_scene, align_patch, get_activation_date, BAND_ORDER
)

# --- Constantes configurables ---
TARGET_GSD = 0.5
TILE_SIZE = 224
BUFFER_METERS = 200

@dataclass
class Scene:
    scene_id: str
    date: object
    geometry: object
    path: Path
    bands: dict
    gsd: float
    cloud_cover: float
    sat_off_nadir: float

@dataclass
class TileMeta:
    fid: int
    tile_path: str
    damaged: int
    has_pre: int
    has_post: int
    split: str
    event: str
    gpkg: str
    shift_x: float
    shift_y: float
    crs_checked: bool
    time_diff_h: float

class Preprocessor:
    def __init__(self, training_root, output_root, percentiles):
        self.training_root = Path(training_root)
        self.output_root = Path(output_root)
        self.percentiles = percentiles

    def load_gpkg(self, gpkg_path):
        gdf = load_gpkg_with_fiona(gpkg_path)
        if gdf.crs is None:
            raise ValueError(f"GPKG {gpkg_path} no tiene CRS")
        # Normaliza columnas
        if 'fid' not in gdf.columns:
            gdf['fid'] = gdf.index
        if 'damaged' not in gdf.columns:
            gdf['damaged'] = -1
        return gdf[['fid', 'damaged', 'geometry']]

    def find_scenes(self, aoi_geom, event_dir, interval_start=None, interval_end=None):
        return find_scenes_for_aoi(aoi_geom, event_dir, interval_start, interval_end)

    def pair_scenes_for_feature(self, scenes, geom, crs, activation_date, printed_scenes=None):
        from preprocess_lib import pick_scene_for_geom
        pre = pick_scene_for_geom(scenes, geom, crs, True, activation_date, printed_scenes=printed_scenes)
        post = pick_scene_for_geom(scenes, geom, crs, False, activation_date, printed_scenes=printed_scenes)
        return pre, post

    def crop_patch(self, scene, bbox, target_crs, is_pre, geom, mask=None, tgt_gsd=None):
        import preprocess_lib
        preprocess_lib.P = self.percentiles
        return process_scene(scene, bbox, target_crs, is_pre, geom, fixed_mask=mask, tgt_gsd=tgt_gsd)

    def align_pre_to_post(self, pre, post, mask=None):
        return align_patch(pre, post, mask=mask)

    def save_tile(self, pre_data, post_data, mask, out_path, meta):
        np.savez_compressed(out_path, pre=pre_data, post=post_data, mask=mask)

    def process_gpkg(self, gpkg_path, split='full'):
        gpkg_path = Path(gpkg_path)
        city_area = gpkg_path.stem.replace('_Training', '').replace('_training', '').replace('_Testing', '').replace('_testing', '').lower()
        output_dir = self.output_root / split / city_area
        output_dir.mkdir(parents=True, exist_ok=True)
        gdf = self.load_gpkg(gpkg_path)
        crs_checked = gdf.crs is not None
        # Limpieza de geometrías antes de reproyectar y unir
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
        gdf['geometry'] = gdf['geometry'].buffer(0)
        aoi_geom_wgs = gdf.to_crs(4326).geometry.union_all()
        event_name = gpkg_path.parent.name
        call_dir = self.training_root / event_name
        activation_date = get_activation_date(call_dir)
        scenes = self.find_scenes(aoi_geom_wgs, call_dir)
        # Añadir logs de fechas de terremoto, mejor PRE y mejor POST
        from preprocess_lib import select_best_scenes
        pre_scene_global, post_scene_global = select_best_scenes(scenes, activation_date)
        print(f"Fecha de activación (terremoto): {activation_date}")
        print(f"Fecha mejor escena PRE: {pre_scene_global['date'] if pre_scene_global else 'N/A'}")
        print(f"Fecha mejor escena POST: {post_scene_global['date'] if post_scene_global else 'N/A'}")
        print(f"Nombre escena PRE: {pre_scene_global['scene_id'] if pre_scene_global else 'N/A'}")
        print(f"Nombre escena POST: {post_scene_global['scene_id'] if post_scene_global else 'N/A'}\n")
        meta_data = []
        printed_scenes = set()  # <-- Nuevo: para controlar los prints
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"Procesando {gpkg_path.name}"):
            fid = row['fid']
            geom = row['geometry']
            # Solo para el primer FID, imprime centroid y bbox en 4326
            if idx == 0:
                from shapely.ops import transform as sxf
                from pyproj import Transformer
                geom_4326 = sxf(Transformer.from_crs(gdf.crs, 4326, always_xy=True).transform, geom)
                buffered_4326 = sxf(Transformer.from_crs(gdf.crs, 4326, always_xy=True).transform, buffer_geom(geom, gdf.crs))
                from preprocess_lib import select_best_scenes
                pre_scene_global, _ = select_best_scenes(scenes, activation_date)
            from preprocess_lib import select_best_scenes
            pre_scene_global, _ = select_best_scenes(scenes, activation_date)
            pre_scene, post_scene = self.pair_scenes_for_feature(scenes, geom, gdf.crs, activation_date, printed_scenes=printed_scenes)
            gsd_pre = pre_scene['gsd'] if pre_scene and 'gsd' in pre_scene and pre_scene['gsd'] else None
            gsd_post = post_scene['gsd'] if post_scene and 'gsd' in post_scene and post_scene['gsd'] else None
            sat_pre = pre_scene['properties']['platform'] if pre_scene and 'properties' in pre_scene and 'platform' in pre_scene['properties'] else None
            sat_post = post_scene['properties']['platform'] if post_scene and 'properties' in post_scene and 'platform' in post_scene['properties'] else None
            # Si ambos existen, comparar GSD
            if gsd_pre and gsd_post:
                try:
                    gsd_pre = float(gsd_pre)
                    gsd_post = float(gsd_post)
                    rel_diff = abs(gsd_pre - gsd_post) / max(gsd_pre, gsd_post)
                    if rel_diff > 0.3:
                        print(f"[SKIP] FID {fid}: Diferencia de GSD demasiado grande entre PRE ({gsd_pre}) y POST ({gsd_post}), descartando tile.")
                        continue
                except Exception:
                    pass
            # GSD objetivo: el peor (mayor valor)
            # Solo usar valores válidos y positivos
            gsd_pre_val = gsd_pre if gsd_pre and float(gsd_pre) > 0 else TARGET_GSD
            gsd_post_val = gsd_post if gsd_post and float(gsd_post) > 0 else TARGET_GSD
            tgt_gsd = max(float(gsd_pre_val), float(gsd_post_val))
        else:
            tgt_gsd = TARGET_GSD
            buffer_m = max(BUFFER_METERS, 300)  # Aumenta a 300m si es menor
            buffered_geom = buffer_geom(geom, gdf.crs)
            bbox = buffered_geom.bounds
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            try:
                transform = from_bounds(*bbox, TILE_SIZE, TILE_SIZE)
                mask = rasterio.features.rasterize(
                    [(buffered_geom, 1)],
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                )
            except Exception:
                mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
            # PRE y POST usan el mismo bbox y mask
            if pre_scene:
                pre_data = self.crop_patch(pre_scene, bbox, gdf.crs, True, geom, mask=mask, tgt_gsd=tgt_gsd)
                if pre_data is None:
                    pre_data = np.zeros((len(BAND_ORDER)+1, TILE_SIZE, TILE_SIZE), np.uint16)
            else:
                pre_data = np.zeros((len(BAND_ORDER)+1, TILE_SIZE, TILE_SIZE), np.uint16)
            if post_scene:
                post_data = self.crop_patch(post_scene, bbox, gdf.crs, False, geom, mask=mask, tgt_gsd=tgt_gsd)
                if post_data is None:
                    post_data = np.zeros((len(BAND_ORDER)+1, TILE_SIZE, TILE_SIZE), np.uint16)
            else:
                post_data = np.zeros((len(BAND_ORDER)+1, TILE_SIZE, TILE_SIZE), np.uint16)
            shift_xy = (0.0, 0.0)
            if pre_scene and post_scene and pre_data is not None and post_data is not None:
                pre_data_hwc = pre_data.transpose(1, 2, 0)
                post_data_hwc = post_data.transpose(1, 2, 0)
                aligned_pre_hwc, shift_xy = self.align_pre_to_post(pre_data_hwc, post_data_hwc, mask=mask)
                pre_data = aligned_pre_hwc.transpose(2, 0, 1)
            out_path = output_dir / f"{fid}.npz"
            self.save_tile(pre_data, post_data, mask, out_path, None)
            time_diff_h = 0.0
            if pre_scene and activation_date:
                time_diff_h = abs((pre_scene['date'] - activation_date).total_seconds()) / 3600
            elif post_scene and activation_date:
                time_diff_h = abs((post_scene['date'] - activation_date).total_seconds()) / 3600
            damaged = int(row['damaged']) if row['damaged'] is not None and pd.notna(row['damaged']) else -1
            meta = TileMeta(
                fid=fid,
                tile_path=str(out_path),
                damaged=damaged,
                has_pre=int(pre_scene is not None),
                has_post=int(post_scene is not None),
                split=split,
                event=event_name,
                gpkg=gpkg_path.name,
                shift_x=shift_xy[1],
                shift_y=shift_xy[0],
                crs_checked=crs_checked,
                time_diff_h=time_diff_h
            )
            # Añadir metadatos extra
            meta_dict = asdict(meta)
            meta_dict['gsd_pre'] = gsd_pre
            meta_dict['gsd_post'] = gsd_post
            meta_dict['sat_pre'] = sat_pre
            meta_dict['sat_post'] = sat_post
            meta_data.append(meta_dict)
        return meta_data 