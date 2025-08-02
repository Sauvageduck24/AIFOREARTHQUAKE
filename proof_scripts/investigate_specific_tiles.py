import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import geopandas as gpd
from shapely.geometry import box, Point
import pandas as pd
from datetime import datetime, timezone
import fiona

def investigate_specific_tiles():
    """Investiga específicamente los tiles 1088, 1089, 1090 para entender por qué no tienen escenas PRE"""
    
    # FIDs problemáticos
    problematic_fids = ['1088', '1089', '1090']
    
    # Rutas de los datos
    preprocessed_dir = Path("preprocessed")
    training_dir = Path("charter-eo4ai-etq-challenge-full/Training")
    gpkg_file = training_dir / "Annotations" / "Call_970_Afghanistan" / "Afghanistan_Area1_Training.gpkg"
    
    print("🔍 INVESTIGANDO TILES ESPECÍFICOS: 1088, 1089, 1090")
    print("=" * 60)
    
    # 1. Analizar los tiles NPZ
    print("\n1️⃣ ANÁLISIS DE TILES NPZ:")
    for fid in problematic_fids:
        tile_path = preprocessed_dir / 'full' / 'afghanistan_area1' / f"{fid}.npz"
        if tile_path.exists():
            data = np.load(tile_path)
            pre = data['pre']
            post = data['post']
            
            print(f"\n📁 {fid}.npz:")
            print(f"   PRE: shape={pre.shape}, dtype={pre.dtype}")
            print(f"   POST: shape={post.shape}, dtype={post.dtype}")
            print(f"   PRE todo ceros: {np.all(pre == 0)}")
            print(f"   POST todo ceros: {np.all(post == 0)}")
            
            if not np.all(post == 0):
                print(f"   POST - Rango: [{post.min():.2f}, {post.max():.2f}], Media: {post.mean():.2f}")
        else:
            print(f"❌ No se encontró: {tile_path}")
    
    # 2. Analizar las geometrías de los edificios
    print(f"\n2️⃣ ANÁLISIS DE GEOMETRÍAS:")
    if not gpkg_file.exists():
        print(f"❌ No se encontró: {gpkg_file}")
        return
    
    try:
        gdf = gpd.read_file(gpkg_file)
        print(f"📊 Total de edificios en GPKG: {len(gdf)}")
        
        # Buscar columna FID
        if 'id' in gdf.columns:
            fid_col = 'id'
        elif 'fid' in gdf.columns:
            fid_col = 'fid'
        else:
            print("❌ No se encontró columna id/fid")
            return
        
        # Filtrar edificios problemáticos
        problematic_buildings = gdf[gdf[fid_col].astype(str).isin(problematic_fids)]
        print(f"🏠 Edificios problemáticos encontrados: {len(problematic_buildings)}")
        
        if len(problematic_buildings) > 0:
            print(f"\n📍 UBICACIONES DE EDIFICIOS PROBLEMÁTICOS:")
            for _, building in problematic_buildings.iterrows():
                fid = building[fid_col]
                geom = building.geometry
                
                # Obtener coordenadas del centro
                if geom.geom_type == 'Point':
                    center = geom
                else:
                    center = geom.centroid
                
                print(f"   FID {fid}:")
                print(f"     Tipo: {geom.geom_type}")
                print(f"     Centro: ({center.x:.6f}, {center.y:.6f})")
                print(f"     Área: {geom.area:.8f} grados²")
                
                # Verificar si está en los bordes del área total
                all_bounds = gdf.total_bounds
                print(f"     Bounds totales: {all_bounds}")
                
                # Calcular distancia a los bordes
                min_x, min_y, max_x, max_y = all_bounds
                dist_to_left = center.x - min_x
                dist_to_right = max_x - center.x
                dist_to_bottom = center.y - min_y
                dist_to_top = max_y - center.y
                
                min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                print(f"     Distancia mínima al borde: {min_dist_to_edge:.6f} grados")
                
                if min_dist_to_edge < 0.001:  # Muy cerca del borde
                    print(f"     ⚠️  MUY CERCA DEL BORDE DEL ÁREA")
        
        # 3. Analizar la distribución espacial
        print(f"\n3️⃣ ANÁLISIS DE DISTRIBUCIÓN ESPACIAL:")
        
        # Calcular centro del área total
        total_center = gdf.geometry.unary_union.centroid
        print(f"   Centro del área total: ({total_center.x:.6f}, {total_center.y:.6f})")
        
        # Calcular distancias al centro
        distances_to_center = []
        for _, building in problematic_buildings.iterrows():
            if building.geometry.geom_type == 'Point':
                center = building.geometry
            else:
                center = building.geometry.centroid
            dist = center.distance(total_center)
            distances_to_center.append(dist)
            print(f"   FID {building[fid_col]}: distancia al centro = {dist:.6f} grados")
        
        avg_distance = np.mean(distances_to_center)
        print(f"   Distancia promedio al centro: {avg_distance:.6f} grados")
        
        # Comparar con todos los edificios
        all_distances = []
        for _, building in gdf.iterrows():
            if building.geometry.geom_type == 'Point':
                center = building.geometry
            else:
                center = building.geometry.centroid
            dist = center.distance(total_center)
            all_distances.append(dist)
        
        all_avg_distance = np.mean(all_distances)
        print(f"   Distancia promedio de todos los edificios: {all_avg_distance:.6f} grados")
        
        if avg_distance > all_avg_distance:
            print(f"   ⚠️  Los edificios problemáticos están más lejos del centro que el promedio")
        
        # 4. Verificar si están en una zona específica
        print(f"\n4️⃣ ANÁLISIS DE ZONA:")
        
        # Crear un buffer alrededor de los edificios problemáticos
        problematic_union = problematic_buildings.geometry.unary_union
        buffer_distance = 0.01  # Aproximadamente 1km
        problematic_buffer = problematic_union.buffer(buffer_distance)
        
        # Contar cuántos edificios están en esta zona
        buildings_in_zone = gdf[gdf.geometry.intersects(problematic_buffer)]
        print(f"   Edificios en zona problemática (±{buffer_distance} grados): {len(buildings_in_zone)}")
        
        # Verificar cuántos de estos también tienen problemas
        problematic_in_zone = buildings_in_zone[buildings_in_zone[fid_col].astype(str).isin(problematic_fids)]
        print(f"   Edificios problemáticos en la zona: {len(problematic_in_zone)}")
        
        if len(buildings_in_zone) > len(problematic_in_zone):
            print(f"   ✅ Hay otros edificios en la misma zona que SÍ tienen datos")
        else:
            print(f"   ⚠️  TODOS los edificios en esta zona tienen problemas")
            
    except Exception as e:
        print(f"❌ Error analizando GPKG: {e}")
    
    # 5. Verificar las escenas disponibles
    print(f"\n5️⃣ VERIFICACIÓN DE ESCENAS:")
    
    # Buscar archivos de escenas en el directorio de training
    call_dir = training_dir / "Call_970_Afghanistan"
    if call_dir.exists():
        scene_dirs = [d for d in call_dir.iterdir() if d.is_dir() and d.name.endswith('-calibrated')]
        print(f"   Escenas encontradas: {len(scene_dirs)}")
        
        # Mostrar algunas escenas
        for i, scene_dir in enumerate(scene_dirs[:5]):
            print(f"   {i+1}. {scene_dir.name}")
        if len(scene_dirs) > 5:
            print(f"   ... y {len(scene_dirs) - 5} más")
    else:
        print(f"❌ No se encontró directorio de escenas: {call_dir}")

def main():
    parser = argparse.ArgumentParser(description="Investigar tiles específicos problemáticos")
    args = parser.parse_args()
    
    investigate_specific_tiles()

if __name__ == "__main__":
    main() 