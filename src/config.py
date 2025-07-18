"""
Configuración del proyecto AI for Earthquake Response.

Este módulo define todos los parámetros configurables para el entrenamiento,
evaluación y predicción de modelos de detección de daños por terremotos.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import dataclasses


@dataclass
class DataConfig:
    """Configuración relacionada con datos y preprocesamiento."""
    
    # Rutas de datos
    data_root: str = "data"
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    splits_path: str = "data/splits"
    
    # Configuración de imágenes
    image_size: int = 512
    channels: int = 3
    num_classes: int = 2  # Dañado vs No dañado
    
    # Configuración de DataLoader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Configuración de splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Configuración de aumentos
    use_augmentation: bool = True
    augmentation_prob: float = 0.5


@dataclass
class ModelConfig:
    """Configuración relacionada con el modelo."""
    
    # Arquitectura
    model_name: str = "unet"
    backbone: str = "resnet50"
    pretrained: bool = True
    
    # Hiperparámetros del modelo
    in_channels: int = 6  # 3 pre + 3 post
    out_channels: int = 1  # Máscara binaria
    encoder_depth: int = 5
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 16])
    
    # Configuración de activaciones
    activation: str = "sigmoid"  # Para clasificación binaria
    
    # Dropout y regularización
    dropout: float = 0.1
    attention: bool = False
    
    # Número de características de entrada para modelos lineales
    in_features: int = 1572864      # Número de características de entrada para modelos lineales
    # Número de salidas para modelos lineales o custom
    out_features: int = 1     # Número de salidas para modelos lineales o custom


@dataclass
class TrainingConfig:
    """Configuración relacionada con el entrenamiento."""
    
    # Configuración básica
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizador
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # Configuración de scheduler
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    warmup_epochs: int = 5
    
    # Configuración de pérdida
    loss_function: str = "bce_dice"  # bce, dice, bce_dice, focal
    class_weights: Optional[List[float]] = None
    
    # Configuración de validación
    val_frequency: int = 1
    save_frequency: int = 10
    
    # Configuración de early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Configuración de checkpointing
    save_best_only: bool = True
    save_last: bool = True


@dataclass
class HardwareConfig:
    """Configuración relacionada con hardware y distribución."""
    
    # Configuración de GPU
    device: str = "auto"  # auto, cuda, cpu
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Configuración de entrenamiento distribuido
    distributed: bool = False
    backend: str = "nccl"  # nccl, gloo
    world_size: int = 1
    rank: int = 0
    
    # Configuración de mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Configuración de memoria
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class LoggingConfig:
    """Configuración relacionada con logging y experimentos."""
    
    # Configuración de experimentos
    experiment_name: str = "earthquake_damage_detection"
    run_name: Optional[str] = None
    
    # Configuración de logging
    log_dir: str = "outputs/logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "ai4eo-earthquake"
    wandb_entity: Optional[str] = None
    
    # Configuración de guardado
    output_dir: str = "outputs"
    models_dir: str = "outputs/models"
    predictions_dir: str = "outputs/predictions"
    
    # Configuración de logging de métricas
    log_frequency: int = 10
    log_images: bool = True
    log_images_frequency: int = 50


@dataclass
class Config:
    """Configuración principal del proyecto."""
    
    # Subconfiguraciones
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Configuración general
    seed: int = 42
    deterministic: bool = True
    debug: bool = False
    
    # Configuración de reproducción
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    
    def __post_init__(self):
        """Validación y configuración post-inicialización."""
        # Asegurar que las rutas existen
        for path in [self.data.data_root, self.logging.output_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Configurar device automáticamente
        if self.hardware.device == "auto":
            import torch
            self.hardware.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configurar run_name si no está especificado
        if self.logging.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logging.run_name = f"{self.model.model_name}_{self.model.backbone}_{timestamp}"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Cargar configuración desde archivo YAML."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Guardar configuración en archivo YAML."""
        config_dict = dataclasses.asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def save(self, output_dir: str):
        """Guardar configuración en el directorio de salida."""
        config_path = Path(output_dir) / "config.yaml"
        self.to_yaml(str(config_path))
        return config_path


def get_default_config() -> Config:
    """Obtener configuración por defecto."""
    return Config()


def get_config_from_args(args) -> Config:
    """Crear configuración desde argumentos de línea de comandos."""
    config = get_default_config()
    
    # Actualizar configuración desde argumentos
    if hasattr(args, 'config') and args.config:
        config = Config.from_yaml(args.config)
    
    # Sobrescribir con argumentos de línea de comandos
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    return config 