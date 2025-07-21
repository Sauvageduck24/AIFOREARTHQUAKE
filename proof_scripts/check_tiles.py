import numpy as np
import os

# Verificar tiles en china
tiles_dir = 'preprocessed/test/china'
tiles = [f for f in os.listdir(tiles_dir) if f.endswith('.npz')][:10]

print("Verificando tiles en china:")
for t in tiles:
    data = np.load(os.path.join(tiles_dir, t))
    pre_max = data['pre'].max()
    post_max = data['post'].max()
    print(f"{t}: pre_max={pre_max:.3f}, post_max={post_max:.3f}")

# Contar tiles con datos válidos
valid_tiles = 0
total_tiles = len([f for f in os.listdir(tiles_dir) if f.endswith('.npz')])

for t in os.listdir(tiles_dir):
    if t.endswith('.npz'):
        data = np.load(os.path.join(tiles_dir, t))
        if data['pre'].max() > 0 or data['post'].max() > 0:
            valid_tiles += 1

print(f"\nResumen: {valid_tiles}/{total_tiles} tiles tienen datos válidos (>0)") 