import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob

# Ruta base
BASE_DIR = Path("preprocessed")

# Subcarpetas donde buscar
SUBSETS = ['full/china', 'test/china']

# Buscar todos los tiles
tile_paths = []
for subset in SUBSETS:
    full_path = BASE_DIR / subset
    tile_paths += list(full_path.glob("*.npz"))

print(f"Encontrados {len(tile_paths)} tiles")

# Función para mostrar un tile
def mostrar_tile(npz_path):
    data = np.load(npz_path)
    pre = data['pre'].astype(np.float32) / 65535
    post = data['post'].astype(np.float32) / 65535

    # Prevenir errores si solo hay 1 canal
    if pre.shape[0] < 3 or post.shape[0] < 3:
        print(f"Tile {npz_path.name} no tiene suficientes bandas para RGB")
        return

    mask = post[3] if post.shape[0] > 3 else np.zeros((224, 224))
    pre_rgb = np.stack([pre[0], pre[1], pre[2]], axis=-1)
    post_rgb = np.stack([post[0], post[1], post[2]], axis=-1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(pre_rgb)
    axs[0].set_title("PRE")
    axs[1].imshow(post_rgb)
    axs[1].set_title("POST")
    axs[2].imshow(post_rgb)
    axs[2].imshow(mask, cmap='Reds', alpha=0.4)
    axs[2].set_title("POST + Mask")
    for ax in axs:
        ax.axis('off')
    plt.suptitle(npz_path.name)
    plt.tight_layout()
    plt.show()

# Navegación interactiva
index = 0
while True:
    if index < 0:
        index = 0
    elif index >= len(tile_paths):
        index = len(tile_paths) - 1

    mostrar_tile(tile_paths[index])

    action = input("\n[n] siguiente | [p] anterior | [q] salir : ").strip().lower()
    if action == 'n':
        index += 1
    elif action == 'p':
        index -= 1
    elif action == 'q':
        break