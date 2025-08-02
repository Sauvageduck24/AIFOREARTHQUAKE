import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_npz_structure(npz_path, verbose=True):
    """Analiza la estructura de un archivo NPZ y muestra información detallada."""
    data = np.load(npz_path)
    
    print(f"\n=== ANÁLISIS DE {npz_path.name} ===")
    
    # Información básica
    print(f"Archivos en NPZ: {list(data.keys())}")
    
    for key in data.keys():
        array = data[key]
        print(f"\n{key.upper()}:")
        print(f"  Forma: {array.shape}")
        print(f"  Tipo: {array.dtype}")
        print(f"  Rango: [{array.min():.2f}, {array.max():.2f}]")
        print(f"  Media: {array.mean():.2f}")
        print(f"  Desv. std: {array.std():.2f}")
        
        if verbose and len(array.shape) == 3:
            # Analizar cada canal por separado
            for i in range(min(5, array.shape[0])):  # Solo los primeros 5 canales
                channel = array[i]
                print(f"    Canal {i}: rango [{channel.min():.2f}, {channel.max():.2f}], media {channel.mean():.2f}")
    
    return data

def mostrar_tile_detallado(npz_path, verbose=False):
    """Muestra un tile con análisis detallado y navegación interactiva."""
    data = np.load(npz_path)
    pre = data['pre'].astype(np.float32) / 65535
    post = data['post'].astype(np.float32) / 65535

    # Prevenir errores si solo hay 1 canal
    if pre.shape[0] < 3 or post.shape[0] < 3:
        print(f"Tile {npz_path.name} no tiene suficientes bandas para RGB")
        return

    # Extraer RGB para análisis
    pre_rgb = np.stack([pre[0], pre[1], pre[2]], axis=-1)
    post_rgb = np.stack([post[0], post[1], post[2]], axis=-1)

    # Mostrar información detallada si verbose está activado
    if verbose:
        print(f"\n=== ANÁLISIS DETALLADO DE {npz_path.name} ===")
        print(f"PRE shape: {pre.shape}, POST shape: {post.shape}")
        
        # Analizar canales PRE
        print(f"\nPRE - Canales disponibles: {pre.shape[0]}")
        for i in range(min(5, pre.shape[0])):
            channel = pre[i]
            print(f"  Canal {i}: rango [{channel.min():.3f}, {channel.max():.3f}], media {channel.mean():.3f}")
        
        # Analizar canales POST
        print(f"\nPOST - Canales disponibles: {post.shape[0]}")
        for i in range(min(5, post.shape[0])):
            channel = post[i]
            print(f"  Canal {i}: rango [{channel.min():.3f}, {channel.max():.3f}], media {channel.mean():.3f}")

    # Extraer máscaras
    pre_mask = pre[3] if pre.shape[0] > 3 else np.zeros((224, 224))
    post_mask = post[3] if post.shape[0] > 3 else np.zeros((224, 224))
    
    if verbose:
        print(f"\nRGB - PRE rango: [{pre_rgb.min():.3f}, {pre_rgb.max():.3f}]")
        print(f"RGB - POST rango: [{post_rgb.min():.3f}, {post_rgb.max():.3f}]")

    # Crear figura con múltiples subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # PRE - RGB
    axs[0, 0].imshow(pre_rgb)
    axs[0, 0].set_title("PRE - RGB")
    axs[0, 0].axis('off')
    
    # POST - RGB
    axs[0, 1].imshow(post_rgb)
    axs[0, 1].set_title("POST - RGB")
    axs[0, 1].axis('off')
    
    # POST + Mask
    axs[0, 2].imshow(post_rgb)
    axs[0, 2].imshow(post_mask, cmap='Reds', alpha=0.4)
    axs[0, 2].set_title("POST + Mask")
    axs[0, 2].axis('off')
    
    # Histograma PRE RGB
    for i, color in enumerate(['red', 'green', 'blue']):
        axs[1, 0].hist(pre_rgb[:,:,i].flatten(), bins=50, alpha=0.7, label=color, color=color)
    axs[1, 0].set_title("Histograma PRE RGB")
    axs[1, 0].legend()
    axs[1, 0].set_xlabel("Valor normalizado")
    axs[1, 0].set_ylabel("Frecuencia")
    
    # Histograma POST RGB
    for i, color in enumerate(['red', 'green', 'blue']):
        axs[1, 1].hist(post_rgb[:,:,i].flatten(), bins=50, alpha=0.7, label=color, color=color)
    axs[1, 1].set_title("Histograma POST RGB")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("Valor normalizado")
    axs[1, 1].set_ylabel("Frecuencia")
    
    # Información de canales
    extra_info = f"Canales extra: PRE={pre.shape[0]-4}, POST={post.shape[0]-4}"
    extra_info += f"\nPRE rango: [{pre_rgb.min():.3f}, {pre_rgb.max():.3f}]"
    extra_info += f"\nPOST rango: [{post_rgb.min():.3f}, {post_rgb.max():.3f}]"
    
    axs[1, 2].text(0.5, 0.5, extra_info, ha='center', va='center', transform=axs[1, 2].transAxes, fontsize=9)
    axs[1, 2].set_title("Información de Canales")
    axs[1, 2].axis('off')
    
    plt.suptitle(f"{npz_path.name} - Análisis Detallado")
    plt.tight_layout()
    plt.show()

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
    parser = argparse.ArgumentParser(description="Visualizar tiles NPZ con análisis detallado")
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada')
    parser.add_argument('--subset', type=str, choices=['full/china', 'test/china'], help='Subcarpeta específica a visualizar')
    args = parser.parse_args()
    
    # Ruta base
    BASE_DIR = Path("preprocessed")
    
    # Subcarpetas donde buscar
    SUBSETS = ['test/china']
    
    # Buscar todos los tiles
    tile_paths = []
    subsets = [args.subset] if args.subset else SUBSETS
    
    for subset in subsets:
        full_path = BASE_DIR / subset
        if full_path.exists():
            tile_paths += list(full_path.glob("*.npz"))
    
    if not tile_paths:
        print("No se encontraron tiles NPZ")
        return
    
    print(f"Encontrados {len(tile_paths)} tiles")
    print("🔍 Analizando calidad de los tiles...")
    
    # Analizar calidad de todos los tiles
    correct_tiles = []
    defective_tiles = []
    
    for tile_path in tile_paths:
        if check_tile_quality(tile_path):
            correct_tiles.append(tile_path)
        else:
            defective_tiles.append(tile_path)
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS DE CALIDAD:")
    print(f"✅ Tiles correctos: {len(correct_tiles)}")
    print(f"❌ Tiles defectuosos: {len(defective_tiles)}")
    print(f"📈 Porcentaje correcto: {(len(correct_tiles)/len(tile_paths)*100):.1f}%")
    
    if defective_tiles:
        print(f"\n🔍 Tiles defectuosos encontrados:")
        for tile in defective_tiles[:10]:  # Mostrar solo los primeros 10
            print(f"  - {tile.name}")
        if len(defective_tiles) > 10:
            print(f"  ... y {len(defective_tiles) - 10} más")
    
    # Usar solo tiles correctos para la visualización
    tile_paths = correct_tiles
    print(f"\n✅ Visualización con análisis detallado de datos preprocesados (solo tiles correctos)")

    # Navegación interactiva
    index = 0
    while True:
        if index < 0:
            index = 0
        elif index >= len(tile_paths):
            index = len(tile_paths) - 1

        mostrar_tile_detallado(tile_paths[index], verbose=args.verbose)

        action = input(f"\n[n] siguiente | [p] anterior | [q] salir (tile {index+1}/{len(tile_paths)}): ").strip().lower()
        if action == 'n':
            index += 1
        elif action == 'p':
            index -= 1
        elif action == 'q':
            break

if __name__ == "__main__":
    main() 