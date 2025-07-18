import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Uso: python calc_max_fids_per_tile.py <ruta_a_meta.json>")
        sys.exit(1)
    meta_path = Path(sys.argv[1])
    if not meta_path.exists():
        print(f"El archivo {meta_path} no existe.")
        sys.exit(1)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    max_fids = 0
    for tile in meta:
        n_fids = len(tile.get('fids', []))
        if n_fids > max_fids:
            max_fids = n_fids
    print(f"Máximo número de fids por tile: {max_fids}")

if __name__ == "__main__":
    main() 