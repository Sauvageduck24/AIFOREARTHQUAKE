import os
import sys
import argparse
import torch
import geopandas as gpd
import numpy as np
import shutil
import json
import fiona
import warnings
from pathlib import Path
from model import get_model
from model import SiameseCNNClassifier
from config import Config
from utils import setup_logging
from data_loader import EarthquakeTileDataset, custom_collate, get_dataloaders
from transforms import get_val_test_transforms
from tqdm import tqdm

# Configurar PROJ para evitar warnings
os.environ['PROJ_LIB'] = os.path.join(os.path.dirname(sys.executable), 'Library', 'share', 'proj')
os.environ['GDAL_DATA'] = os.path.join(os.path.dirname(sys.executable), 'Library', 'share', 'gdal')

# Suprimir warnings específicos de pyproj
warnings.filterwarnings("ignore", message="pyproj unable to set PROJ database path")

# --- Argumentos ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generar predicciones para submission de la competición.")
    parser.add_argument('--model_path', type=str, default='outputs/models/checkpoint_epoch_2.pth', help='Ruta al modelo .pth')
    parser.add_argument('--input_dir', type=str, default='submissions/submission_1', help='Carpeta con archivos .gpkg a predecir')
    parser.add_argument('--output_dir', type=str, default='submissions/submission_5', help='Carpeta donde guardar la submission')
    parser.add_argument('--config', type=str, default='config.yaml', help='Archivo de configuración')
    parser.add_argument('--device', type=str, default='cuda', help='cpu o cuda')
    parser.add_argument('--split', type=str, default='test', help='Split de preprocesado a usar (test o full)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size para inferencia')
    parser.add_argument('--threshold', type=float, default=None, help='Umbral para binarización (override config)')
    parser.add_argument('--num_workers', type=int, default=4, help='Número de workers para DataLoader')
    parser.add_argument('--prueba', action='store_true',default=True, help='Modo prueba: solo procesa 1 batch para verificar funcionamiento')
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

def load_gpkg_with_fiona(gpkg_path):
    """Carga un archivo GPKG usando Fiona y lo convierte a GeoDataFrame."""
    features = []
    feature_ids = []
    selected_fid_column = None  # Guardar qué columna usamos
    crs = None
    
    with fiona.open(gpkg_path) as src:
        crs = src.crs
        schema_columns = list(src.schema['properties'].keys())
        
        # Buscar columnas que empiecen con 'fid_'
        fid_columns = [col for col in schema_columns if col.lower().startswith('fid_')]
        
        if fid_columns:
            # Si encontramos columnas fid_*, usar la primera como identificador
            selected_fid_column = fid_columns[0]
            
            for feature in src:
                features.append(feature)
                # Usar el valor de la columna fid_* como identificador
                fid_value = feature['properties'].get(selected_fid_column)
                feature_ids.append(fid_value)
        else:
            # Si no hay columnas fid_*, usar el id del feature de Fiona
            for feature in src:
                features.append(feature)
                # Extraer el id del feature de Fiona
                feature_ids.append(feature.get('id'))
    
    # Convertir a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    
    # Asignar los IDs a la columna apropiada
    if selected_fid_column:
        # Si usamos una columna fid_*, mantener ese nombre
        gdf[selected_fid_column] = feature_ids
    else:
        # Si usamos el ID de Fiona, asignar a columna 'id'
        gdf['id'] = feature_ids
    
    return gdf

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Cargar configuración y modelo
    config = Config.from_yaml(args.config)
    config.model.model_name = 'siamese_cnn_classifier'  # Forzar el modelo correcto para el checkpoint
    config.hardware.device = args.device
    device = torch.device(args.device)
    # Usar threshold óptimo guardado en config si existe
    threshold = getattr(config.training, 'threshold', None)
    if threshold is None:
        threshold = args.threshold if args.threshold is not None else 0.5
    max_channels = getattr(config.data, 'max_channels', 3)
    # Dataset de test
    # Usar get_dataloaders para obtener el test_loader con los mismos parámetros que en train
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.split = args.split
    train_loader, val_loader, test_loader = get_dataloaders(config)
    if test_loader is None:
        raise RuntimeError('No se pudo crear el test_loader con get_dataloaders')
    # Cargar modelo
    model = get_model(config.model)
    model = load_checkpoint_to_model(model, args.model_path, device)
    model.eval()
    model.to(device)
    # --- Inferencia por batches ---
    pred_dict = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Inferencia')): 
            # Si es modo prueba, solo procesar 1 batch
            if args.prueba and i >= 1:
                print(f"[PRUEBA] Procesado 1 batch. Saltando el resto...")
                break
                
            images = batch['image'].to(device)  # [B, C, H, W]
            building_mask = batch['building_mask'].to(device) if 'building_mask' in batch else None
            if isinstance(model, SiameseCNNClassifier):
                logits = model(images, building_mask)
            else:
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
            for j in range(len(preds)):
                pred_dict[(events[j], fids[j])] = int(preds[j])
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
        gdf = load_gpkg_with_fiona(in_path)
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
        
        # Si es modo prueba, mostrar información sobre las predicciones
        if args.prueba:
            total_fids = len(damages)
            fids_con_prediccion = sum(1 for d in damages if d != 0)
            print(f"[PRUEBA] {fname}: {fids_con_prediccion}/{total_fids} FIDs tienen predicción (resto = 0)")
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
    
    if args.prueba:
        print(f"[PRUEBA] Submission de prueba generada en: {args.output_dir}, {zip_base}.zip")
        print("[PRUEBA] Modo prueba completado. Para ejecución completa, omitir --prueba")
    else:
        print(f"Submission generada en: {args.output_dir}, {zip_base}.zip")

    # Mostrar el head de cada archivo .gpkg guardado en la carpeta de salida
    print("\n[INFO] Contenido de los archivos .gpkg generados:")
    for fname in sorted(os.listdir(args.output_dir)):
        if fname.endswith('.gpkg'):
            out_path = os.path.join(args.output_dir, fname)
            gdf = load_gpkg_with_fiona(out_path)
            # Mostrar recuento de damaged=0 y damaged=1
            count_0 = (gdf['damaged'] == 0).sum()
            count_1 = (gdf['damaged'] == 1).sum()
            print(f"\nArchivo: {fname}")
            print(f"Recuento damaged=0: {count_0} | damaged=1: {count_1}")
            print(gdf.head())

if __name__ == '__main__':
    main()
