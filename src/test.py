import os
import sys
import argparse
import torch
import geopandas as gpd
import numpy as np
import shutil
import json
from pathlib import Path
from model import get_model
from config import Config
from utils import setup_logging
from transforms import get_val_test_transforms
from tqdm import tqdm

# --- Argumentos ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generar predicciones para submission de la competición.")
    parser.add_argument('--model_path', type=str, default='outputs/models/checkpoint_epoch_2.pth', help='Ruta al modelo .pth')
    parser.add_argument('--input_dir', type=str, default='submissions/submission_1', help='Carpeta con archivos .gpkg a predecir')
    parser.add_argument('--output_dir', type=str, default='submissions/submission_2', help='Carpeta donde guardar la submission')
    parser.add_argument('--config', type=str, default='config.yaml', help='Archivo de configuración')
    parser.add_argument('--device', type=str, default='cuda', help='cpu o cuda')
    parser.add_argument('--split', type=str, default='test', help='Split de preprocesado a usar (test o full)')
    return parser.parse_args()

def load_checkpoint_to_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Cargar configuración y modelo
    config = Config.from_yaml(args.config)
    config.hardware.device = args.device
    model = get_model(config.model)
    model = load_checkpoint_to_model(model, args.model_path, args.device)
    model.eval()
    model.to(args.device)
    # --- Leer meta.json ---
    preproc_dir = f"data/preprocessed/{args.split}"
    meta_path = os.path.join(preproc_dir, 'meta.json')
    if not os.path.exists(meta_path):
        preproc_dir = "data/preprocessed/full"
        meta_path = os.path.join(preproc_dir, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    # Crear un índice: (tile_id) -> meta_entry
    tileid_to_meta = {entry['id']: entry for entry in meta}
    # --- Predecir por tile ---
    # Guardar predicciones: (tile_id, fid) -> prediccion
    pred_dict = {}
    for tile_id, entry in tileid_to_meta.items():
        img_path = os.path.join(preproc_dir, 'images', f"{tile_id}.npz")
        if not os.path.exists(img_path):
            continue
        img_npz = np.load(img_path)
        image = img_npz[img_npz.files[0]]  # (C, H, W)
        if image.shape[0] > 6:
            image = image[:6]  # Solo los primeros 6 canales
        image = torch.from_numpy(image).float().unsqueeze(0).to(args.device)  # [1, C, H, W]
        # DEBUG: imprimir shapes
        #print(f"Tile: {tile_id}")
        #print('  image.shape:', image.shape)
        #if hasattr(model, 'in_features'):
            #print('  Esperado por el modelo (in_features):', model.in_features)
            #print('  C*H*W:', image.numel() // image.shape[0])
        with torch.no_grad():
            logits = model(image)
            if logits.ndim == 2:
                logits = logits.squeeze(0)
            else:
                logits = logits[0]
            # Si logits es mayor que 1D, tomar la salida binaria
            if logits.ndim > 1:
                logits = logits.squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            # Binarizar (umbral 0.5)
            preds = (probs > 0.5).astype(int)
        fids = entry.get('fids', [])
        for i, fid in enumerate(fids):
            pred_dict[(tile_id, fid)] = int(preds[i]) if i < len(preds) else 0
    # --- Procesar cada archivo .gpkg ---
    gpkg_files = [f for f in os.listdir(args.input_dir) if f.endswith('.gpkg')]
    for fname in tqdm(gpkg_files, desc='Procesando archivos .gpkg'):
        in_path = os.path.join(args.input_dir, fname)
        gdf = gpd.read_file(in_path)
        # Inferir nombre base para tile_id
        base = os.path.splitext(fname)[0]
        # Buscar columna fid_<base> (ignorando mayúsculas/minúsculas)
        fid_col = None
        for col in gdf.columns:
            if col.lower().startswith('fid_') and base.lower() in col.lower():
                fid_col = col
                break
        # Si no existe, buscar la primera columna que empiece por fid_
        if fid_col is None:
            for col in gdf.columns:
                if col.lower().startswith('fid_'):
                    fid_col = col
                    break
        # Si tampoco, buscar columna llamada fid
        if fid_col is None and 'fid' in gdf.columns:
            fid_col = 'fid'
        if fid_col is None:
            raise ValueError(f"No se encontró columna fid_<basename>, ni ninguna columna que empiece por fid_ ni 'fid' en {fname}")
        damages = []
        # Filtrar tiles de meta.json que contengan el nombre base
        candidate_tiles = [tid for tid in tileid_to_meta if base in tid]
        if not candidate_tiles:
            print(f"[ADVERTENCIA] No se encontraron tiles en meta.json que contengan '{base}' para {fname}. Se asigna damage=0 a todas las filas.")
            gdf['damage'] = 0
            out_path = os.path.join(args.output_dir, fname)
            gdf.to_file(out_path, driver='GPKG')
            continue
        tile_id = candidate_tiles[0]  # Si hay varios, toma el primero (puedes mejorar esto si hace falta)
        for idx, row in gdf.iterrows():
            fid = row[fid_col]
            pred = pred_dict.get((tile_id, fid), 0)
            damages.append(pred)
        gdf['damage'] = damages
        out_path = os.path.join(args.output_dir, fname)
        gdf.to_file(out_path, driver='GPKG')
    # Comprimir la carpeta de submission
    zip_base = os.path.abspath(args.output_dir)
    shutil.make_archive(zip_base, 'zip', args.output_dir)
    print(f"Submission generada en: {args.output_dir}, {zip_base}.zip")

if __name__ == '__main__':
    main()
