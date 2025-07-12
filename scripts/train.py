#!/usr/bin/env python3
"""
Script principal de entrenamiento para AI for Earthquake Response.

Este script ejecuta el entrenamiento completo de modelos de detección
de daños por terremotos usando imágenes satelitales multitemporales.
"""

import argparse
import sys
from pathlib import Path

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config, get_default_config
from src.train import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo de detección de daños por terremotos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Entrenamiento básico con configuración por defecto
  python scripts/train.py

  # Entrenamiento con archivo de configuración personalizado
  python scripts/train.py --config configs/unet_resnet50.yaml

  # Entrenamiento con parámetros específicos
  python scripts/train.py --epochs 100 --batch_size 16 --lr 0.001

  # Resumir entrenamiento desde checkpoint
  python scripts/train.py --resume outputs/models/checkpoint_epoch_50.pth

  # Entrenamiento distribuido
  python scripts/train.py --distributed --gpu_ids 0 1 2 3

  # Entrenamiento con múltiples seeds
  python scripts/train.py --seeds 42 123 456
        """
    )
    
    # Configuración del experimento
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML")
    parser.add_argument("--resume", type=str, help="Ruta al checkpoint para resumir entrenamiento")
    parser.add_argument("--seed", type=int, help="Semilla para reproducibilidad")
    parser.add_argument("--experiment_name", type=str, help="Nombre del experimento")
    
    # Parámetros de datos
    parser.add_argument("--data_root", type=str, help="Ruta raíz de los datos")
    parser.add_argument("--image_size", type=int, help="Tamaño de las imágenes")
    parser.add_argument("--batch_size", type=int, help="Tamaño de batch")
    parser.add_argument("--num_workers", type=int, help="Número de workers para DataLoader")
    
    # Parámetros del modelo
    parser.add_argument("--model", type=str, help="Nombre del modelo (unet, resnet_unet, efficientnet_unet, transformer)")
    parser.add_argument("--backbone", type=str, help="Backbone para modelos basados en ResNet/EfficientNet")
    parser.add_argument("--pretrained", action="store_true", help="Usar pesos pre-entrenados")
    
    # Parámetros de entrenamiento
    parser.add_argument("--epochs", type=int, help="Número de épocas")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], help="Optimizador")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "step", "plateau"], help="Scheduler")
    parser.add_argument("--loss_function", type=str, choices=["bce", "dice", "bce_dice", "focal"], help="Función de pérdida")
    
    # Parámetros de hardware
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], help="Dispositivo a usar")
    parser.add_argument("--gpu_ids", type=int, nargs="+", help="IDs de GPUs a usar")
    parser.add_argument("--distributed", action="store_true", help="Usar entrenamiento distribuido")
    parser.add_argument("--use_amp", action="store_true", help="Usar mixed precision training")
    
    # Parámetros de logging
    parser.add_argument("--output_dir", type=str, help="Directorio de salida")
    parser.add_argument("--tensorboard", action="store_true", help="Usar TensorBoard")
    parser.add_argument("--wandb", action="store_true", help="Usar Weights & Biases")
    parser.add_argument("--wandb_project", type=str, help="Nombre del proyecto en W&B")
    parser.add_argument("--wandb_entity", type=str, help="Entidad de W&B")
    
    # Parámetros de validación
    parser.add_argument("--val_frequency", type=int, help="Frecuencia de validación")
    parser.add_argument("--save_frequency", type=int, help="Frecuencia de guardado")
    parser.add_argument("--early_stopping_patience", type=int, help="Paciencia para early stopping")
    
    # Parámetros de debug
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    parser.add_argument("--deterministic", action="store_true", help="Modo determinístico")
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    # Cargar configuración
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Sobrescribir configuración con argumentos de línea de comandos
    if args.seed is not None:
        config.seed = args.seed
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name
    
    # Datos
    if args.data_root is not None:
        config.data.raw_data_path = args.data_root
    if args.image_size is not None:
        config.data.image_size = args.image_size
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    
    # Modelo
    if args.model is not None:
        config.model.model_name = args.model
    if args.backbone is not None:
        config.model.backbone = args.backbone
    if args.pretrained:
        config.model.pretrained = True
    
    # Entrenamiento
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.optimizer is not None:
        config.training.optimizer = args.optimizer
    if args.scheduler is not None:
        config.training.scheduler = args.scheduler
    if args.loss_function is not None:
        config.training.loss_function = args.loss_function
    
    # Hardware
    if args.device is not None:
        config.hardware.device = args.device
    if args.gpu_ids is not None:
        config.hardware.gpu_ids = args.gpu_ids
    if args.distributed:
        config.hardware.distributed = True
    if args.use_amp:
        config.hardware.use_amp = True
    
    # Logging
    if args.output_dir is not None:
        config.logging.output_dir = args.output_dir
    if args.tensorboard:
        config.logging.tensorboard = True
    if args.wandb:
        config.logging.wandb = True
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.logging.wandb_entity = args.wandb_entity
    
    # Validación
    if args.val_frequency is not None:
        config.training.val_frequency = args.val_frequency
    if args.save_frequency is not None:
        config.training.save_frequency = args.save_frequency
    if args.early_stopping_patience is not None:
        config.training.early_stopping_patience = args.early_stopping_patience
    
    # Debug
    if args.debug:
        config.debug = True
    if args.deterministic:
        config.deterministic = True
    
    # Crear trainer y ejecutar entrenamiento
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main() 