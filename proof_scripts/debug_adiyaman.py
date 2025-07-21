import sys
import os
sys.path.append('src')

from preprocess import find_scenes_for_aoi, select_best_scenes, get_activation_date
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

if not adiyaman_gpkg:
    print("No se encontró GPKG de adiyaman")
    exit(1)

print(f"GPKG encontrado: {adiyaman_gpkg}")

# Leer datos
gdf = gpd.read_file(adiyaman_gpkg)
print(f"Edificios en GPKG: {len(gdf)}")

# Limpiar geometrías
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
gdf['geometry'] = gdf['geometry'].buffer(0)
print(f"Edificios tras limpieza: {len(gdf)}")

# Verificar CRS
if gdf.crs is None:
    print("⚠️ GPKG no tiene CRS")
    gdf.set_crs(4326, inplace=True)

# AOI para intersección
aoi_geom_wgs = gdf.to_crs(4326).geometry.union_all()
print(f"AOI bounds: {aoi_geom_wgs.bounds}")

# Obtener evento y fecha de activación
event_name = adiyaman_gpkg.parent.name
call_dir = training_data / event_name
activation_date = get_activation_date(call_dir)
print(f"Evento: {event_name}")
print(f"Fecha de activación: {activation_date}")

# Encontrar escenas
scenes = find_scenes_for_aoi(aoi_geom_wgs, call_dir)
print(f"Escenas encontradas: {len(scenes)}")

if scenes:
    for i, scene in enumerate(scenes):
        print(f"  Escena {i+1}: {scene['scene_id']}")
        print(f"    Fecha: {scene['date']}")
        print(f"    Bandas: {list(scene['bands'].keys())}")
        print(f"    Bounds: {scene['geometry'].bounds}")
    
    # Seleccionar mejores escenas
    pre_scene, post_scene = select_best_scenes(scenes, activation_date)
    print(f"\nMejores escenas:")
    print(f"  Pre: {pre_scene['scene_id'] if pre_scene else 'None'}")
    print(f"  Post: {post_scene['scene_id'] if post_scene else 'None'}")
else:
    print("❌ No se encontraron escenas para adiyaman") 