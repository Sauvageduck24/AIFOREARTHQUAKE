import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description="Visualiza un tile pre/post de los archivos .npz generados"
)
parser.add_argument('--tile', type=str, default="preprocessed/full/china/0.npz", help='Ruta al archivo .npz del tile')
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

print(f"pre shape: {pre_tile.shape}, dtype: {pre_tile.dtype}")
print(f"post shape: {post_tile.shape}, dtype: {post_tile.dtype}")

# Diagnóstico de los datos
for name, tile in [('pre', pre_tile), ('post', post_tile)]:
    if tile.shape[0] >= 3:
        rgb = np.transpose(tile[:3], (1,2,0))
        if tile.dtype == np.uint16:
            rgb = rgb / 65535.0
        print(f"{name} RGB min: {rgb.min()}, max: {rgb.max()}, mean: {rgb.mean(axis=(0,1))}")
    else:
        print(f"{name} tiene menos de 3 canales")

# Visualización PRE, POST, DIF y máscaras
show_diff = pre_tile.shape[0] >= 3 and post_tile.shape[0] >= 3
show_mask = pre_tile.shape[0] >= 4 or post_tile.shape[0] >= 4
n = 2 + int(show_diff)
fig, axs = plt.subplots(2, n, figsize=(5*n, 8))
idx = 0
# Fila 1: RGB
for key, img in zip(['PRE','POST'], [pre_tile, post_tile]):
    if img.shape[0] >= 3:
        rgb = np.transpose(img[:3], (1,2,0))
        if img.dtype == np.uint16:
            rgb = rgb / 65535.0
        axs[0, idx].set_title(key)
        axs[0, idx].imshow((rgb * 255).astype(np.uint8))
        axs[0, idx].axis('off')
        idx += 1
# Diferencia absoluta
if show_diff:
    pre_rgb = np.transpose(pre_tile[:3], (1,2,0))
    post_rgb = np.transpose(post_tile[:3], (1,2,0))
    if pre_tile.dtype == np.uint16:
        pre_rgb = pre_rgb / 65535.0
    if post_tile.dtype == np.uint16:
        post_rgb = post_rgb / 65535.0
    diff = np.abs(post_rgb - pre_rgb)
    axs[0, idx].set_title('DIF |POST - PRE|')
    axs[0, idx].imshow((diff * 255).astype(np.uint8))
    axs[0, idx].axis('off')
# Fila 2: Máscaras
idx = 0
pre_mask = pre_tile[3] if pre_tile.shape[0] >= 4 else None
post_mask = post_tile[3] if post_tile.shape[0] >= 4 else None
for key, mask in zip(['Máscara PRE','Máscara POST'], [pre_mask, post_mask]):
    if mask is not None:
        axs[1, idx].set_title(key)
        axs[1, idx].imshow(mask, cmap='gray')
        axs[1, idx].axis('off')
    else:
        axs[1, idx].set_title(f"{key} (no disp.)")
        axs[1, idx].axis('off')
    idx += 1
# Diferencia de máscaras
if pre_mask is not None and post_mask is not None and n > 2:
    mask_diff = np.abs(post_mask.astype(np.int16) - pre_mask.astype(np.int16))
    axs[1, idx].set_title('DIF Máscaras')
    axs[1, idx].imshow(mask_diff, cmap='hot')
    axs[1, idx].axis('off')
plt.tight_layout()
plt.show()