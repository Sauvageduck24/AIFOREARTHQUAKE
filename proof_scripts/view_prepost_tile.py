import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def normalize_for_display(img, method='percentile'):
    """Mejorada para imágenes con valores muy dispersos"""
    img = img.astype(np.float32)
    
    # Si la imagen es uniforme, devolver como está
    if img.min() == img.max():
        return img.astype(np.uint8)
    
    if method == 'percentile':
        # Usar percentiles para evitar outliers
        if img.max() > 0:
            p2 = np.percentile(img[img > 0], 2)
            p98 = np.percentile(img[img > 0], 98)
            if p98 > p2:
                img = np.clip(img, p2, p98)
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    elif method == 'histogram_equalization':
        # Equalización de histograma para mejorar contraste
        from skimage import exposure
        img = exposure.equalize_hist(img)
    elif method == 'adaptive':
        # Normalización adaptativa más agresiva
        if img.max() > 0:
            p5 = np.percentile(img[img > 0], 5)
            p95 = np.percentile(img[img > 0], 95)
            if p95 > p5:
                img = np.clip(img, p5, p95)
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    img *= 255
    return img.astype(np.uint8)

def create_rgb_image(data, method='standard'):
    """Crear imagen RGB a partir de múltiples bandas con diferentes métodos"""
    if len(data.shape) == 3:
        # Si es (C, H, W), convertir a (H, W, C)
        if data.shape[0] <= 3:
            # Tomar las primeras bandas disponibles
            rgb = data[:3, :, :].transpose(1, 2, 0)
        else:
            if method == 'standard':
                # Usar red, green, blue (índices 2, 1, 0)
                if data.shape[0] >= 4:
                    rgb = data[[2, 1, 0], :, :].transpose(1, 2, 0)
                else:
                    rgb = np.stack([data[0, :, :]] * 3, axis=-1)
            elif method == 'false_color':
                # Combinación falsa para mejor contraste
                if data.shape[0] >= 4:
                    # NIR, Red, Green para vegetación
                    rgb = data[[3, 2, 1], :, :].transpose(1, 2, 0)
                else:
                    rgb = np.stack([data[0, :, :]] * 3, axis=-1)
            elif method == 'enhanced':
                # Combinación mejorada para urbanas
                if data.shape[0] >= 4:
                    # Red, NIR, Blue
                    rgb = data[[2, 3, 0], :, :].transpose(1, 2, 0)
                else:
                    rgb = np.stack([data[0, :, :]] * 3, axis=-1)
    else:
        # Si es 2D, convertir a RGB
        rgb = np.stack([data] * 3, axis=-1)
    
    return rgb

parser = argparse.ArgumentParser(description="Visualiza un tile pre/post de los archivos .npz generados")
parser.add_argument('--tile', type=str, required=True, help='Ruta al archivo .npz del tile')
parser.add_argument('--normalize', type=str, default='percentile', 
                   choices=['percentile', 'histogram_equalization', 'adaptive'],
                   help='Método de normalización')
parser.add_argument('--rgb_method', type=str, default='standard',
                   choices=['standard', 'false_color', 'enhanced'],
                   help='Método de combinación RGB')
args = parser.parse_args()

# Cargar tile
if not os.path.exists(args.tile):
    print(f"El archivo {args.tile} no existe.")
    exit(1)

data = np.load(args.tile, allow_pickle=True)

print("Contenido del archivo .npz:")
for k in data.files:
    print(f"  {k}: {data[k].shape if hasattr(data[k], 'shape') else data[k]}")

# Cargar arrays pre y post
if 'pre' in data.files and 'post' in data.files:
    pre_tile = data['pre']
    post_tile = data['post']
else:
    print("El archivo no contiene arrays 'pre' y 'post'")
    exit(1)

print(f"pre shape: {pre_tile.shape}")
print(f"post shape: {post_tile.shape}")

# Diagnóstico de los datos
print(f"\nDiagnóstico:")
print(f"Pre - min: {pre_tile.min():.6f}, max: {pre_tile.max():.6f}, mean: {pre_tile.mean():.6f}")
print(f"Post - min: {post_tile.min():.6f}, max: {post_tile.max():.6f}, mean: {post_tile.mean():.6f}")

# Visualizar
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Post-evento RGB
if post_tile.size > 0:
    post_rgb = create_rgb_image(post_tile, args.rgb_method)
    post_display = normalize_for_display(post_rgb, args.normalize)
    axs[0,0].imshow(post_display)
    axs[0,0].set_title(f'Post-evento (RGB: {args.rgb_method}, Norm: {args.normalize})')
    axs[0,0].axis('off')
else:
    axs[0,0].set_title('Post-evento (no disponible)')
    axs[0,0].axis('off')

# Pre-evento RGB
if pre_tile.size > 0:
    pre_rgb = create_rgb_image(pre_tile, args.rgb_method)
    pre_display = normalize_for_display(pre_rgb, args.normalize)
    axs[0,1].imshow(pre_display)
    axs[0,1].set_title(f'Pre-evento (RGB: {args.rgb_method}, Norm: {args.normalize})')
    axs[0,1].axis('off')
else:
    axs[0,1].set_title('Pre-evento (no disponible)')
    axs[0,1].axis('off')

# Histogramas
if post_tile.size > 0:
    post_flat = post_tile.flatten()
    axs[1,0].hist(post_flat[post_flat > 0], bins=50, alpha=0.7, color='blue')
    axs[1,0].set_title('Histograma Post-evento (valores > 0)')
    axs[1,0].set_xlabel('Valor')
    axs[1,0].set_ylabel('Frecuencia')

if pre_tile.size > 0:
    pre_flat = pre_tile.flatten()
    axs[1,1].hist(pre_flat[pre_flat > 0], bins=50, alpha=0.7, color='red')
    axs[1,1].set_title('Histograma Pre-evento (valores > 0)')
    axs[1,1].set_xlabel('Valor')
    axs[1,1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show() 