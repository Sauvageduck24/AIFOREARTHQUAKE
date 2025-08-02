import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import geopandas as gpd
from shapely.geometry import box
import pandas as pd

def analyze_afghanistan_tiles():
    """Analiza específicamente los tiles de Afghanistan para entender problemas de cobertura"""
    
    # Rutas de los datos
    preprocessed_dir = Path("preprocessed")
    training_dir = Path("charter-eo4ai-etq-challenge-full/Training")
    
    # Buscar tiles de Afghanistan
    afghanistan_tiles = []
    for area in ['afghanistan_area1_training', 'afghanistan_area2_training']:
        area_dir = preprocessed_dir / 'test' / area
        if area_dir.exists():
            tiles = list(area_dir.glob("*.npz"))
            afghanistan_tiles.extend(tiles)
    
    if not afghanistan_tiles:
        print("No se encontraron tiles de Afghanistan")
        return
    
    print(f"Encontrados {len(afghanistan_tiles)} tiles de Afghanistan")
    
    # Analizar cada tile
    defective_tiles = []
    tiles_without_pre = []
    tiles_without_post = []
    
    for tile_path in afghanistan_tiles:
        try:
            data = np.load(tile_path)
            pre = data['pre']
            post = data['post']
            
            fid = tile_path.stem
            
            # Verificar si son todo ceros
            pre_all_zero = np.all(pre == 0)
            post_all_zero = np.all(post == 0)
            
            if pre_all_zero:
                tiles_without_pre.append(fid)
            if post_all_zero:
                tiles_without_post.append(fid)
            if pre_all_zero or post_all_zero:
                defective_tiles.append({
                    'fid': fid,
                    'pre_all_zero': pre_all_zero,
                    'post_all_zero': post_all_zero,
                    'tile_path': str(tile_path)
                })
                
        except Exception as e:
            print(f"Error analizando {tile_path.name}: {e}")
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS DE TILES DE AFGHANISTAN:")
    print(f"Total de tiles: {len(afghanistan_tiles)}")
    print(f"Tiles defectuosos: {len(defective_tiles)}")
    print(f"Sin escena PRE: {len(tiles_without_pre)}")
    print(f"Sin escena POST: {len(tiles_without_post)}")
    print(f"Porcentaje defectuoso: {(len(defective_tiles)/len(afghanistan_tiles)*100):.1f}%")
    
    if defective_tiles:
        print(f"\n🔍 ANÁLISIS DE TILES DEFECTUOSOS:")
        
        # Contar tipos de defectos
        pre_only = sum(1 for t in defective_tiles if t['pre_all_zero'] and not t['post_all_zero'])
        post_only = sum(1 for t in defective_tiles if not t['pre_all_zero'] and t['post_all_zero'])
        both = sum(1 for t in defective_tiles if t['pre_all_zero'] and t['post_all_zero'])
        
        print(f"  - Solo sin PRE: {pre_only}")
        print(f"  - Solo sin POST: {post_only}")
        print(f"  - Sin ambas: {both}")
        
        # Mostrar algunos ejemplos
        print(f"\n📋 EJEMPLOS DE TILES DEFECTUOSOS:")
        for i, tile in enumerate(defective_tiles[:10]):
            status = []
            if tile['pre_all_zero']:
                status.append("sin PRE")
            if tile['post_all_zero']:
                status.append("sin POST")
            print(f"  {i+1}. {tile['fid']} ({', '.join(status)})")
        
        if len(defective_tiles) > 10:
            print(f"  ... y {len(defective_tiles) - 10} más")
    
    # Analizar las geometrías de los edificios problemáticos
    print(f"\n🗺️  ANALIZANDO GEOMETRÍAS DE EDIFICIOS PROBLEMÁTICOS...")
    
    # Cargar los archivos GPKG originales
    gpkg_files = [
        training_dir / "Annotations" / "Call_970_Afghanistan" / "Afghanistan_Area1_Training.gpkg",
        training_dir / "Annotations" / "Call_970_Afghanistan" / "Afghanistan_Area2_Training.gpkg"
    ]
    
    for gpkg_file in gpkg_files:
        if not gpkg_file.exists():
            print(f"No se encontró: {gpkg_file}")
            continue
            
        print(f"\n📁 Analizando: {gpkg_file.name}")
        
        try:
            gdf = gpd.read_file(gpkg_file)
            print(f"  Total de edificios: {len(gdf)}")
            
            # Encontrar edificios problemáticos
            problematic_fids = [t['fid'] for t in defective_tiles]
            
            # Buscar estos FIDs en el GPKG
            if 'id' in gdf.columns:
                fid_col = 'id'
            elif 'fid' in gdf.columns:
                fid_col = 'fid'
            else:
                print("  No se encontró columna id/fid")
                continue
            
            problematic_buildings = gdf[gdf[fid_col].astype(str).isin(problematic_fids)]
            print(f"  Edificios problemáticos encontrados: {len(problematic_buildings)}")
            
            if len(problematic_buildings) > 0:
                # Analizar la distribución espacial
                bounds = problematic_buildings.total_bounds
                print(f"  Bounds de edificios problemáticos: {bounds}")
                
                # Verificar si están en los bordes del área
                all_bounds = gdf.total_bounds
                print(f"  Bounds totales del área: {all_bounds}")
                
                # Calcular distancia al centro
                center = gdf.geometry.unary_union.centroid
                distances = [b.geometry.distance(center) for b in problematic_buildings.itertuples()]
                avg_distance = np.mean(distances)
                print(f"  Distancia promedio al centro: {avg_distance:.2f} grados")
                
        except Exception as e:
            print(f"  Error analizando {gpkg_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analizar tiles de Afghanistan")
    parser.add_argument('--output', type=str, help='Archivo JSON para guardar resultados')
    args = parser.parse_args()
    
    analyze_afghanistan_tiles()

if __name__ == "__main__":
    main() 