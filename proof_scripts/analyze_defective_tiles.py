import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

def analyze_defective_tile(npz_path):
    """Analiza un tile defectuoso para entender por qué falló"""
    try:
        data = np.load(npz_path)
        pre = data['pre']
        post = data['post']
        
        print(f"\n=== ANÁLISIS DE TILE DEFECTUOSO: {npz_path.name} ===")
        print(f"PRE shape: {pre.shape}, POST shape: {post.shape}")
        print(f"PRE dtype: {pre.dtype}, POST dtype: {post.dtype}")
        
        # Verificar si son todo ceros
        pre_all_zero = np.all(pre == 0)
        post_all_zero = np.all(post == 0)
        pre_all_nan = np.all(np.isnan(pre))
        post_all_nan = np.all(np.isnan(post))
        
        print(f"PRE todo ceros: {pre_all_zero}")
        print(f"POST todo ceros: {post_all_zero}")
        print(f"PRE todo NaN: {pre_all_nan}")
        print(f"POST todo NaN: {post_all_nan}")
        
        # Si no son todo ceros, mostrar estadísticas
        if not pre_all_zero and not pre_all_nan:
            print(f"PRE - Rango: [{pre.min():.2f}, {pre.max():.2f}], Media: {pre.mean():.2f}")
        if not post_all_zero and not post_all_nan:
            print(f"POST - Rango: [{post.min():.2f}, {post.max():.2f}], Media: {post.mean():.2f}")
        
        # Extraer FID del nombre del archivo
        fid = npz_path.stem
        print(f"FID del edificio: {fid}")
        
        return {
            'fid': fid,
            'pre_all_zero': pre_all_zero,
            'post_all_zero': post_all_zero,
            'pre_all_nan': pre_all_nan,
            'post_all_nan': post_all_nan,
            'pre_shape': pre.shape,
            'post_shape': post.shape
        }
        
    except Exception as e:
        print(f"Error al cargar {npz_path.name}: {e}")
        return None

def check_tile_quality(npz_path):
    """Verifica si un tile es defectuoso (pre o post todo 0 o nulo)"""
    try:
        data = np.load(npz_path)
        pre = data['pre']
        post = data['post']
        
        # Verificar si pre o post son todo 0 o nulos
        pre_defective = np.all(pre == 0) or np.all(np.isnan(pre))
        post_defective = np.all(post == 0) or np.all(np.isnan(post))
        
        return not (pre_defective or post_defective)
    except Exception as e:
        print(f"Error al cargar {npz_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Analizar tiles defectuosos")
    parser.add_argument('--subset', type=str, default='test/china', help='Subcarpeta específica a analizar')
    parser.add_argument('--output', type=str, help='Archivo JSON para guardar resultados')
    args = parser.parse_args()
    
    # Ruta base
    BASE_DIR = Path("preprocessed")
    full_path = BASE_DIR / args.subset
    
    if not full_path.exists():
        print(f"No se encontró la carpeta: {full_path}")
        return
    
    # Buscar todos los tiles
    tile_paths = list(full_path.glob("*.npz"))
    
    if not tile_paths:
        print("No se encontraron tiles NPZ")
        return
    
    print(f"Encontrados {len(tile_paths)} tiles en {full_path}")
    print("🔍 Analizando calidad de los tiles...")
    
    # Analizar calidad de todos los tiles
    correct_tiles = []
    defective_tiles = []
    defective_analysis = []
    
    for tile_path in tile_paths:
        if check_tile_quality(tile_path):
            correct_tiles.append(tile_path)
        else:
            defective_tiles.append(tile_path)
            # Analizar el tile defectuoso
            analysis = analyze_defective_tile(tile_path)
            if analysis:
                defective_analysis.append(analysis)
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS DE CALIDAD:")
    print(f"✅ Tiles correctos: {len(correct_tiles)}")
    print(f"❌ Tiles defectuosos: {len(defective_tiles)}")
    print(f"📈 Porcentaje correcto: {(len(correct_tiles)/len(tile_paths)*100):.1f}%")
    
    if defective_tiles:
        print(f"\n🔍 ANÁLISIS DE TILES DEFECTUOSOS:")
        print(f"Total de tiles defectuosos: {len(defective_tiles)}")
        
        # Contar tipos de defectos
        pre_only_zero = sum(1 for a in defective_analysis if a['pre_all_zero'] and not a['post_all_zero'])
        post_only_zero = sum(1 for a in defective_analysis if not a['pre_all_zero'] and a['post_all_zero'])
        both_zero = sum(1 for a in defective_analysis if a['pre_all_zero'] and a['post_all_zero'])
        
        print(f"  - Solo PRE con ceros: {pre_only_zero}")
        print(f"  - Solo POST con ceros: {post_only_zero}")
        print(f"  - Ambos PRE y POST con ceros: {both_zero}")
        
        # Mostrar algunos ejemplos
        print(f"\n📋 EJEMPLOS DE TILES DEFECTUOSOS:")
        for i, tile in enumerate(defective_tiles[:5]):
            print(f"  {i+1}. {tile.name}")
        if len(defective_tiles) > 5:
            print(f"  ... y {len(defective_tiles) - 5} más")
    
    # Guardar resultados si se especifica
    if args.output:
        results = {
            'total_tiles': len(tile_paths),
            'correct_tiles': len(correct_tiles),
            'defective_tiles': len(defective_tiles),
            'defective_analysis': defective_analysis,
            'defective_fids': [a['fid'] for a in defective_analysis]
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main() 