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
from data_loader import EarthquakeTileDataset, custom_collate
from transforms import get_val_test_transforms
from tqdm import tqdm

# --- Argumentos ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generar predicciones para submission de la competición.")
    parser.add_argument('--model_path', type=str, default='outputs/models/checkpoint_epoch_10.pth', help='Ruta al modelo .pth')
    parser.add_argument('--input_dir', type=str, default='submissions/submission_1', help='Carpeta con archivos .gpkg a predecir')
    parser.add_argument('--output_dir', type=str, default='submissions/submission_3', help='Carpeta donde guardar la submission')
    parser.add_argument('--config', type=str, default='config.yaml', help='Archivo de configuración')
    parser.add_argument('--device', type=str, default='cuda', help='cpu o cuda')
    parser.add_argument('--split', type=str, default='test', help='Split de preprocesado a usar (test o full)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size para inferencia')
    parser.add_argument('--threshold', type=float, default=None, help='Umbral para binarización (override config)')
    parser.add_argument('--num_workers', type=int, default=12, help='Número de workers para DataLoader')
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

def get_ciudad_a_evento(annotations_dir):
    ciudad_a_evento = {}
    for evento_folder in os.listdir(annotations_dir):
        evento_path = os.path.join(annotations_dir, evento_folder)
        if os.path.isdir(evento_path):
            for f in os.listdir(evento_path):
                if f.endswith('.gpkg'):
                    base = f.split('_')[0].lower()
                    ciudad_a_evento[base] = evento_folder.lower()
    return ciudad_a_evento

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Cargar configuración y modelo
    config = Config.from_yaml(args.config)
    config.model.model_name = 'cnn_classifier'  # Forzar el modelo correcto para el checkpoint
    config.hardware.device = args.device
    device = torch.device(args.device)
    # Override threshold si se pasa por argumento
    threshold = args.threshold if args.threshold is not None else getattr(config.training, 'threshold', 0.5)
    max_channels = getattr(config.data, 'max_channels', 3)
    # Dataset de test
    data_root = getattr(config.data, 'data_root', 'preprocessed')
    test_dataset = EarthquakeTileDataset(
        data_root=data_root,
        split=args.split,
        mode=getattr(config.data, 'mode', 'post_only'),
        filter_has_data=getattr(config.data, 'filter_has_data', True),
        max_channels=max_channels
    )
    # DataLoader para test
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    # Cargar modelo
    model = get_model(config.model)
    model = load_checkpoint_to_model(model, args.model_path, device)
    model.eval()
    model.to(device)
    # --- Inferencia por batches ---
    pred_dict = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inferencia'): 
            images = batch['image'].to(device)  # [B, C, H, W]
            logits = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, 1] o [B]
            # Si probs es [B,1], aplanar
            if probs.ndim == 2 and probs.shape[1] == 1:
                probs = probs[:, 0]
            preds = (probs > threshold).astype(int)
            # DEBUG: Mostrar predicciones y probabilidades del batch
            #print(f"[DEBUG] Batch probs: {probs[:20]}")
            #print(f"[DEBUG] Batch preds: {preds[:20]}")
            fids = batch['fid']
            events = batch['event']
            for i in range(len(preds)):
                pred_dict[(events[i], fids[i])] = int(preds[i])
    # DEBUG: Mostrar ejemplos de claves en pred_dict
    print(f"[DEBUG] Ejemplo de claves en pred_dict: {list(pred_dict.keys())[:20]}")
    # --- Procesar cada archivo .gpkg ---
    gpkg_files = [f for f in os.listdir(args.input_dir) if f.endswith('.gpkg')]
    eventos_pred = set([k[0] for k in pred_dict.keys()])
    print(f"[DEBUG] Eventos disponibles en pred_dict: {eventos_pred}")
    # Obtener mapeo dinámico ciudad->evento
    annotations_dir = "data/international-charter-eo4ai-etq-challenge/Training/Annotations"
    ciudad_a_evento = get_ciudad_a_evento(annotations_dir)
    for fname in tqdm(gpkg_files, desc='Procesando archivos .gpkg'):
        in_path = os.path.join(args.input_dir, fname)
        gdf = gpd.read_file(in_path)
        # Extraer nombre base de ciudad/evento del nombre del archivo
        base = os.path.splitext(fname)[0]
        # Tomar la parte de ciudad/evento (antes de '_testing')
        event_name = base.split('_')[0]
        # Buscar columna fid_<base> (ignorando mayúsculas/minúsculas)
        fid_col = None
        for col in gdf.columns:
            if col.lower().startswith('fid_') and base.lower() in col.lower():
                fid_col = col
                break
        if fid_col is None:
            for col in gdf.columns:
                if col.lower().startswith('fid_'):
                    fid_col = col
                    break
        if fid_col is None and 'fid' in gdf.columns:
            fid_col = 'fid'
        if fid_col is None:
            raise ValueError(f"No se encontró columna fid_<basename>, ni ninguna columna que empiece por fid_ ni 'fid' en {fname}")
        damages = []
        for idx, row in gdf.iterrows():
            fid = int(row[fid_col])  # Asegurar tipo int
            ciudad = event_name.lower()
            evento_ref = ciudad_a_evento.get(ciudad, ciudad)
            posibles_eventos = [e for e in eventos_pred if ciudad in e.lower()]
            if posibles_eventos:
                evento_usar = posibles_eventos[0]
            else:
                posibles_eventos = [e for e in eventos_pred if evento_ref in e.lower()]
                if posibles_eventos:
                    evento_usar = posibles_eventos[0]
                else:
                    evento_usar = event_name
            pred = pred_dict.get((evento_usar, fid), 0)
            damages.append(pred)
        # DEBUG: Mostrar ejemplos de predicciones y conteo de daños
        #print(f"[DEBUG] {fname} - Ejemplo de predicciones:", damages[:20])
        #print(f"[DEBUG] {fname} - Conteo de daños:", sum(damages))
        # Crear nuevo DataFrame solo con fid_*, damaged (predicho) y geometry
        output_gdf = gdf[[fid_col, 'geometry']].copy()
        output_gdf['damaged'] = damages
        # Reordenar columnas: fid_*, damaged, geometry
        output_gdf = output_gdf[[fid_col, 'damaged', 'geometry']]
        out_path = os.path.join(args.output_dir, fname)
        output_gdf.to_file(out_path, driver='GPKG')
    # Comprimir la carpeta de submission
    zip_base = os.path.abspath(args.output_dir)
    shutil.make_archive(zip_base, 'zip', args.output_dir)
    print(f"Submission generada en: {args.output_dir}, {zip_base}.zip")

    # Mostrar el head de cada archivo .gpkg guardado en la carpeta de salida
    print("\n[INFO] Contenido de los archivos .gpkg generados:")
    for fname in sorted(os.listdir(args.output_dir)):
        if fname.endswith('.gpkg'):
            out_path = os.path.join(args.output_dir, fname)
            gdf = gpd.read_file(out_path)
            # Mostrar recuento de damaged=0 y damaged=1
            count_0 = (gdf['damaged'] == 0).sum()
            count_1 = (gdf['damaged'] == 1).sum()
            print(f"\nArchivo: {fname}")
            print(f"Recuento damaged=0: {count_0} | damaged=1: {count_1}")
            print(gdf.head())

if __name__ == '__main__':
    main()
