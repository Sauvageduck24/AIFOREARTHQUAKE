import sys
import os
sys.path.append('src')

from preprocess import find_scenes_for_aoi, select_best_scenes, get_activation_date, process_scene, buffer_geom
import geopandas as gpd
from pathlib import Path
import numpy as np

# Configurar paths
data_root = Path("data/international-charter-eo4ai-etq-challenge")
training_annotations = data_root / "Training" / "Annotations"
training_data = data_root / "Training"

# Encontrar el GPKG de adiyaman
adiyaman_gpkg = None
for root, _, files in os.walk(training_annotations):
    for file in files:
        if 'adiyaman' in file.lower() and file.endswith('.gpkg'):
            adiyaman_gpkg = Path(root) / file
            break
    if adiyaman_gpkg:
        break

print(f"GPKG encontrado: {adiyaman_gpkg}")

# Leer datos
gdf = gpd.read_file(adiyaman_gpkg)
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
gdf['geometry'] = gdf['geometry'].buffer(0)

if gdf.crs is None:
    gdf.set_crs(4326, inplace=True)

# Tomar el primer edificio
building = gdf.iloc[0].geometry
print(f"Edificio seleccionado: {gdf.iloc[0]['fid_adiyaman']}")
print(f"Geometría bounds: {building.bounds}")

# AOI para intersección
aoi_geom_wgs = gdf.to_crs(4326).geometry.union_all()

# Obtener evento y fecha de activación
event_name = adiyaman_gpkg.parent.name
call_dir = training_data / event_name
activation_date = get_activation_date(call_dir)

# Encontrar escenas
scenes = find_scenes_for_aoi(aoi_geom_wgs, call_dir)
print(f"Escenas encontradas: {len(scenes)}")

if scenes:
    # Seleccionar mejores escenas
    pre_scene, post_scene = select_best_scenes(scenes, activation_date)
    print(f"Pre scene: {pre_scene['scene_id'] if pre_scene else 'None'}")
    print(f"Post scene: {post_scene['scene_id'] if post_scene else 'None'}")
    
    # Procesar un edificio
    print("\n" + "="*50)
    print("PROCESANDO EDIFICIO INDIVIDUAL")
    print("="*50)
    
    # Buffer consistente
    buffered_geom = buffer_geom(building, gdf.crs)
    bbox = buffered_geom.bounds
    print(f"Bbox después del buffer: {bbox}")
    
    # Procesar escena pre
    if pre_scene:
        print("\n--- PROCESANDO ESCENA PRE ---")
        pre_data = process_scene(pre_scene, bbox, gdf.crs)
        print(f"Resultado pre: {type(pre_data)}, shape: {pre_data.shape if pre_data is not None else 'None'}")
        if pre_data is not None:
            print(f"Valores pre: min={pre_data.min()}, max={pre_data.max()}, dtype={pre_data.dtype}")
    else:
        print("No hay escena pre")
    
    # Procesar escena post
    if post_scene:
        print("\n--- PROCESANDO ESCENA POST ---")
        post_data = process_scene(post_scene, bbox, gdf.crs)
        print(f"Resultado post: {type(post_data)}, shape: {post_data.shape if post_data is not None else 'None'}")
        if post_data is not None:
            print(f"Valores post: min={post_data.min()}, max={post_data.max()}, dtype={post_data.dtype}")
    else:
        print("No hay escena post")
else:
    print("❌ No se encontraron escenas para adiyaman") 