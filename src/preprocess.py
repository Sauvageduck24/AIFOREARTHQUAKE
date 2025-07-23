import argparse
import shutil
import sys
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pipeline import Preprocessor
from preprocess_lib import (
    logger,
    sample_patches_for_percentiles,
    PERCENTILE_SAMPLE_SIZE
)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def normalize_for_display(img):
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img

def main():
    import logging
    global P
    # --- Redirige stdout y stderr a outputs/salida_consola.txt y consola ---
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    log_path = outputs_dir / "salida_consola.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    # --- Logging a archivo y consola ---
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limitar número de GPKGs a procesar')
    parser.add_argument('--loglevel', type=str, default='INFO', help='Nivel de logging (DEBUG, INFO, WARNING)')
    parser.add_argument('--review', action='store_true',default=False, help='Mostrar plot de ejemplo al final de cada GPKG')
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    output_root = Path("preprocessed")
    if output_root.exists() and output_root.is_dir():
        shutil.rmtree(output_root)
    data_root = Path("data/international-charter-eo4ai-etq-challenge")
    training_annotations = data_root / "Training" / "Annotations"
    training_data = data_root / "Training"
    output_root.mkdir(exist_ok=True)
    logger.info("Calculando percentiles globales por banda...")
    P = sample_patches_for_percentiles(training_data, sample_size=PERCENTILE_SAMPLE_SIZE)
    logger.info(f"Percentiles globales: {P}")
    gpkg_files = []
    for root, _, files in os.walk(training_annotations):
        for file in files:
            if file.lower().endswith(('training.gpkg', 'testing.gpkg')):
                gpkg_files.append(Path(root) / file)
    logger.info(f"Encontrados {len(gpkg_files)} archivos GPKG para procesar")
    for gpkg in gpkg_files:
        logger.info(f"  - {gpkg.name}")
    if args.limit:
        gpkg_files = gpkg_files[:args.limit]

    gpkg_files=gpkg_files[1:2]  # Elimina este filtro para procesar todos los seleccionados

    all_meta = []
    pre = Preprocessor(training_data, output_root, percentiles=P)
    for gpkg_path in gpkg_files:
        fname = gpkg_path.name.lower()
        scene_name = gpkg_path.stem
        print(f"\n[INFO] Procesando escena: {scene_name}")
        if fname.endswith("training.gpkg"):
            split = 'full'
        elif fname.endswith("testing.gpkg"):
            split = 'test'
        else:
            split = 'unknown'
        meta = pre.process_gpkg(gpkg_path, split=split)
        all_meta.extend(meta)

        # Visualización de un ejemplo de tile generado solo si --review
        if args.review:
            try:
                last_gpkg = gpkg_files[0]
                split = 'full' if last_gpkg.name.lower().endswith('training.gpkg') else 'test'
                city_area = last_gpkg.stem.replace('_Training', '').replace('_training', '').replace('_Testing', '').replace('_testing', '').lower()
                npz_dir = output_root / split / city_area
                npz_files = sorted(glob.glob(str(npz_dir / '*.npz')))
                if npz_files:
                    npz_path = npz_files[0]
                    print(f"npz_path: {npz_path}")
                    arr = np.load(npz_path)
                    pre_img = arr['pre']
                    post = arr['post']
                    plt.figure(figsize=(10,4))
                    n = 0
                    if pre_img.shape[0] >= 3:
                        n += 1
                    if post.shape[0] >= 3:
                        n += 1
                    idx = 1
                    for key, img in zip(['PRE','POST'], [pre_img, post]):
                        if img.shape[0] >= 3:
                            rgb = np.transpose(img[:3], (1,2,0))
                            if img.dtype == np.uint16:
                                rgb = rgb / 65535.0
                            plt.subplot(1, n, idx)
                            plt.title(key)
                            plt.imshow((rgb * 255).astype(np.uint8))
                            plt.axis('off')
                            idx += 1
                    plt.show()
                else:
                    print('[INFO] No se encontró ningún archivo .npz para visualizar.')
            except Exception as e:
                print(f'[ERROR] No se pudo visualizar el tile de ejemplo: {e}')

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(output_root / "meta.csv", index=False)
    logger.info(f"Procesamiento completo! Guardados {len(all_meta)} tiles")
    log_file.close()

if __name__ == "__main__":
    main() 