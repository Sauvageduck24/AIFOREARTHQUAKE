import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def analyze_npz(npz_path):
    arr = np.load(npz_path)
    for key in arr.files:
        print(f"[INFO] Key: {key}, shape: {arr[key].shape}, dtype: {arr[key].dtype}")
    # Analiza 'pre' y 'post' si existen
    rgb_imgs = {}
    for key in ['pre', 'post']:
        if key in arr:
            data = arr[key]
            print(f"\n[INFO] Análisis del canal '{key}': shape={data.shape}, dtype={data.dtype}")
            for i in range(data.shape[0]):
                ch = data[i]
                print(f"  Canal {i}: min={ch.min()}, max={ch.max()}, mean={ch.mean()} (uint16)")
            # Visualiza RGB si hay al menos 3 canales
            if data.shape[0] >= 3:
                rgb = np.transpose(data[:3], (1,2,0)) / 65535.0
                print(f"  [INFO] RGB preview: min={rgb.min()}, max={rgb.max()}, mean={rgb.mean(axis=(0,1))}")
                rgb_imgs[key] = rgb
            else:
                print(f"  [WARN] Menos de 3 canales, no se puede mostrar RGB.")
    # Mostrar pre y post juntos si ambos existen
    if 'pre' in rgb_imgs or 'post' in rgb_imgs:
        n = sum([k in rgb_imgs for k in ['pre','post']])
        plt.figure(figsize=(5*n,5))
        idx = 1
        for key in ['pre','post']:
            if key in rgb_imgs:
                plt.subplot(1, n, idx)
                plt.title(f"{key.upper()} RGB Preview")
                plt.imshow((rgb_imgs[key] * 255).astype(np.uint8))
                plt.axis('off')
                idx += 1
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_npz_channels.py ruta/al/archivo.npz")
        sys.exit(1)
    npz_path = Path(sys.argv[1])
    if not npz_path.exists():
        print(f"No existe el archivo: {npz_path}")
        sys.exit(1)
    analyze_npz(npz_path) 