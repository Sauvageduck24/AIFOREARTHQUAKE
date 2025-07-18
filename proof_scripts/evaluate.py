#!/usr/bin/env python3
"""
Script de evaluación para AI for Earthquake Response.

Este script evalúa modelos entrenados en el conjunto de test
y genera reportes detallados de rendimiento.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from loguru import logger

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config, get_default_config
from src.model import get_model
from src.data_loader import get_data_loaders
from src.utils import (
    set_seed, setup_logging, calculate_metrics, load_checkpoint,
    plot_confusion_matrix, create_confusion_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluación de modelo de detección de daños por terremotos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Evaluación básica
  python scripts/evaluate.py --model_path outputs/models/best_model.pth

  # Evaluación con configuración específica
  python scripts/evaluate.py --model_path outputs/models/best_model.pth --config configs/unet_resnet50.yaml

  # Evaluación con datos específicos
  python scripts/evaluate.py --model_path outputs/models/best_model.pth --data_path data/processed/test

  # Evaluación con múltiples umbrales
  python scripts/evaluate.py --model_path outputs/models/best_model.pth --thresholds 0.3 0.5 0.7

  # Evaluación con TTA
  python scripts/evaluate.py --model_path outputs/models/best_model.pth --use_tta
        """
    )
    
    # Parámetros obligatorios
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al modelo a evaluar")
    
    # Configuración
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML")
    parser.add_argument("--data_path", type=str, help="Ruta a los datos de test")
    parser.add_argument("--output_dir", type=str, help="Directorio de salida para resultados")
    
    # Parámetros de evaluación
    parser.add_argument("--batch_size", type=int, help="Tamaño de batch para evaluación")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5], help="Umbrales para binarización")
    parser.add_argument("--use_tta", action="store_true", help="Usar Test Time Augmentation")
    parser.add_argument("--save_predictions", action="store_true", help="Guardar predicciones")
    
    # Parámetros de hardware
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], help="Dispositivo a usar")
    parser.add_argument("--num_workers", type=int, help="Número de workers para DataLoader")
    
    # Parámetros de logging
    parser.add_argument("--verbose", action="store_true", help="Modo verbose")
    parser.add_argument("--save_plots", action="store_true", help="Guardar gráficos")
    
    return parser.parse_args()


def evaluate_model(model, test_loader, config, device, thresholds=[0.5], use_tta=False):
    """
    Evaluar modelo en conjunto de test.
    
    Args:
        model: Modelo a evaluar
        test_loader: DataLoader para datos de test
        config: Configuración del experimento
        device: Dispositivo a usar
        thresholds: Lista de umbrales para evaluar
        use_tta: Si usar Test Time Augmentation
        
    Returns:
        Diccionario con resultados de evaluación
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    logger.info("Iniciando evaluación...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)
            
            if use_tta:
                # Test Time Augmentation
                predictions = []
                for i in range(4):  # 4 transformaciones
                    # Aplicar transformación
                    if i == 0:
                        # Original
                        pred = model(images)
                    elif i == 1:
                        # Flip horizontal
                        pred = model(torch.flip(images, [3]))
                        pred = torch.flip(pred, [3])
                    elif i == 2:
                        # Flip vertical
                        pred = model(torch.flip(images, [2]))
                        pred = torch.flip(pred, [2])
                    elif i == 3:
                        # Rotate 180
                        pred = model(torch.rot90(images, k=2, dims=[2, 3]))
                        pred = torch.rot90(pred, k=2, dims=[2, 3])
                    
                    predictions.append(pred)
                
                # Promedio de predicciones
                predictions = torch.stack(predictions).mean(dim=0)
            else:
                predictions = model(images)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # Guardar algunas imágenes para visualización
            if batch_idx == 0:
                all_images.append(images.cpu())
            
            if batch_idx % 10 == 0:
                logger.info(f"Procesados {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenar todos los resultados
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    logger.info(f"Evaluación completada. Forma de predicciones: {all_predictions.shape}")
    
    # Evaluar con diferentes umbrales
    results = {}
    for threshold in thresholds:
        logger.info(f"Evaluando con umbral {threshold}...")
        
        metrics = calculate_metrics(all_predictions, all_targets, threshold=threshold)
        results[f"threshold_{threshold}"] = metrics
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'images': torch.cat(all_images, dim=0) if all_images else None,
        'results': results,
        'thresholds': thresholds
    }


def generate_evaluation_report(eval_results, config, model_path, output_dir):
    """
    Generar reporte de evaluación.
    
    Args:
        eval_results: Resultados de evaluación
        config: Configuración del experimento
        model_path: Ruta al modelo evaluado
        output_dir: Directorio de salida
    """
    logger.info("Generando reporte de evaluación...")
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Resultados principales (usando umbral 0.5 por defecto)
    main_results = eval_results['results'].get('threshold_0.5', {})
    
    # Crear reporte
    report = {
        'model_path': str(model_path),
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'config': {
            'model_name': config.model.model_name,
            'backbone': config.model.backbone,
            'image_size': config.data.image_size,
            'batch_size': config.data.batch_size,
            'thresholds': eval_results['thresholds']
        },
        'metrics': main_results,
        'all_thresholds': eval_results['results']
    }
    
    # Guardar reporte JSON
    report_path = output_path / "evaluation_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Crear tabla de métricas
    df_metrics = pd.DataFrame.from_dict(eval_results['results'], orient='index')
    df_metrics.index = [f"Threshold {t}" for t in eval_results['thresholds']]
    
    # Guardar tabla CSV
    metrics_path = output_path / "metrics.csv"
    df_metrics.to_csv(metrics_path)
    
    # Mostrar resultados principales
    logger.info("=== RESULTADOS DE EVALUACIÓN ===")
    logger.info(f"Modelo: {config.model.model_name}")
    if config.model.backbone:
        logger.info(f"Backbone: {config.model.backbone}")
    logger.info(f"Umbral: 0.5")
    logger.info(f"Accuracy: {main_results['accuracy']:.4f}")
    logger.info(f"Precision: {main_results['precision']:.4f}")
    logger.info(f"Recall: {main_results['recall']:.4f}")
    logger.info(f"F1-Score: {main_results['f1_score']:.4f}")
    logger.info(f"IoU: {main_results['iou']:.4f}")
    logger.info(f"Dice: {main_results['dice']:.4f}")
    logger.info(f"AUC-ROC: {main_results['auc_roc']:.4f}")
    logger.info(f"AUC-PR: {main_results['auc_pr']:.4f}")
    
    # Guardar matriz de confusión
    if eval_results['predictions'] is not None and eval_results['targets'] is not None:
        cm = create_confusion_matrix(eval_results['predictions'], eval_results['targets'])
        cm_path = output_path / "confusion_matrix.png"
        plot_confusion_matrix(cm, save_path=str(cm_path))
        
        # Guardar matriz como CSV
        cm_df = pd.DataFrame(cm, 
                           columns=['Predicted No Damage', 'Predicted Damage'],
                           index=['Actual No Damage', 'Actual Damage'])
        cm_csv_path = output_path / "confusion_matrix.csv"
        cm_df.to_csv(cm_csv_path)
    
    logger.info(f"Reporte guardado en: {output_path}")
    
    return report


def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar logging
    setup_logging()
    
    # Cargar configuración
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Sobrescribir configuración
    if args.data_path:
        config.data.raw_data_path = args.data_path
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device:
        config.hardware.device = args.device
    if args.num_workers:
        config.data.num_workers = args.num_workers
    
    # Configurar semilla
    set_seed(config.seed, config.deterministic)
    
    # Configurar dispositivo
    device = torch.device(config.hardware.device)
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    logger.info(f"Cargando modelo desde: {args.model_path}")
    model = get_model(config.model)
    model = model.to(device)
    
    # Cargar checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Modelo cargado desde checkpoint (época {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Modelo cargado (solo pesos)")
    
    # Cargar datos de test
    logger.info("Cargando datos de test...")
    _, _, test_loader = get_data_loaders(config)
    
    # Evaluar modelo
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
        thresholds=args.thresholds,
        use_tta=args.use_tta
    )
    
    # Generar reporte
    output_dir = args.output_dir or config.logging.output_dir
    report = generate_evaluation_report(
        eval_results=eval_results,
        config=config,
        model_path=args.model_path,
        output_dir=output_dir
    )
    
    # Guardar predicciones si se solicita
    if args.save_predictions:
        predictions_path = Path(output_dir) / "predictions.npy"
        targets_path = Path(output_dir) / "targets.npy"
        
        np.save(predictions_path, eval_results['predictions'].numpy())
        np.save(targets_path, eval_results['targets'].numpy())
        
        logger.info(f"Predicciones guardadas en: {predictions_path}")
        logger.info(f"Targets guardados en: {targets_path}")
    
    logger.info("Evaluación completada exitosamente")


if __name__ == "__main__":
    main() 