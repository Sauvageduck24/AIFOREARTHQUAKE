"""
Módulo de entrenamiento para el proyecto AI for Earthquake Response.

Este módulo implementa el ciclo completo de entrenamiento y validación
para modelos de detección de daños por terremotos.

Ahora soporta cualquier modelo de PyTorch, incluyendo modelos simples como nn.Linear:

Ejemplo de uso con modelo lineal:
  - En config.yaml:
      model:
        model_name: "linear"
        in_features: 8  # canales*H*W si imagen completa
        out_features: 1 # binario
  - O por línea de comandos:
      python src/train.py --model linear
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from loguru import logger
import wandb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

console = Console()

from config import Config, get_default_config
# from data_loader import PreprocessedTileDataset  # Eliminado porque no existe ni se usa
from model import get_model
from utils import (
    set_seed, setup_logging, calculate_metrics, calculate_loss,
    save_checkpoint, load_checkpoint, log_metrics, setup_wandb,
    visualize_predictions, plot_metrics_history
)


def get_preprocessed_dataloaders(config):
    """
    Crea DataLoaders usando PreprocessedTileDataset para los splits disponibles.
    """
    from torch.utils.data import DataLoader
    data_root = config.data.processed_data_path if hasattr(config.data, 'processed_data_path') else 'data/preprocessed'
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    pin_memory = config.data.pin_memory
    splits = ['full', 'val', 'test']
    loaders = {}
    for split in splits:
        try:
            dataset = PreprocessedTileDataset(data_root=data_root, split=split, transform=None)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='full'), num_workers=num_workers, pin_memory=pin_memory)
            loaders[split] = loader
        except Exception as e:
            loaders[split] = None
    train_loader = loaders.get('full', None)
    val_loader = loaders.get('val', None)
    test_loader = loaders.get('test', None)
    return train_loader, val_loader, test_loader

class Trainer:
    """
    Clase principal para el entrenamiento de modelos.
    
    Maneja el ciclo completo de entrenamiento, validación y logging.
    """
    
    def __init__(self, config: Config):
        """
        Inicializar el trainer.
        
        Args:
            config: Configuración del experimento
        """
        self.config = config
        self.device = torch.device(config.hardware.device)
        
        # Configurar semilla
        set_seed(config.seed, config.deterministic)
        
        # Configurar logging
        setup_logging(config.logging.log_dir)
        
        # Configurar W&B
        if config.logging.wandb:
            setup_wandb(
                project_name=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                config=vars(config),
                tags=["earthquake", "damage_detection", config.model.model_name]
            )
        
        # Configurar TensorBoard
        self.tensorboard_writer = None
        if config.logging.tensorboard:
            self.tensorboard_writer = SummaryWriter(config.logging.log_dir)
        
        # Inicializar componentes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Historial de métricas
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1_score': [],
            'val_f1_score': [],
            'train_iou': [],
            'val_iou': []
        }
        
        # Mejor métrica para early stopping
        self.best_metric = 0.0
        self.patience_counter = 0
        
        self.mode = getattr(config.data, 'mode', 'basic') if hasattr(config, 'data') else 'basic'
        logger.info("Trainer inicializado")
        # Variables para guardar el mejor modelo en memoria
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_epoch = 0
        self.best_train_metrics = None
        self.best_val_metrics = None
    
    def setup_data(self):
        """Cargar DataLoader de train y val desde disco."""
        mode = getattr(self.config.data, 'mode', 'basic') if hasattr(self.config, 'data') else 'basic'
        train_path = f"data/dataloaders/dataloader_train_{mode}.pt"
        val_path = f"data/dataloaders/dataloader_val_{mode}.pt"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"No se encontró el dataloader de train en {train_path}. Ejecuta src/data_loader.py primero.")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"No se encontró el dataloader de val en {val_path}. Ejecuta src/data_loader.py primero.")
        self.train_loader = torch.load(train_path)
        self.val_loader = torch.load(val_path)
        print(f"[INFO] DataLoader de train cargado desde {train_path} ({len(self.train_loader)} batches)")
        print(f"[INFO] DataLoader de val cargado desde {val_path} ({len(self.val_loader)} batches)")
    def setup_model(self):
        """Configurar modelo, optimizador y scheduler."""
        logger.info("Configurando modelo...")
        
        # Crear modelo
        self.model = get_model(self.config.model)
        self.model = self.model.to(self.device)
        
        # Configurar optimizador
        if self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Optimizador {self.config.training.optimizer} no soportado")
        
        # Configurar scheduler
        if self.config.training.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                verbose=True
            )
        
        # Configurar mixed precision
        if self.config.hardware.use_amp:
            self.scaler = GradScaler()
        
        logger.info(f"Modelo configurado: {self.config.model.model_name}")
        logger.info(f"Optimizador: {self.config.training.optimizer}")
        logger.info(f"Scheduler: {self.config.training.scheduler}")
    
    def setup_distributed(self):
        """Configurar entrenamiento distribuido."""
        if self.config.hardware.distributed:
            # Configurar proceso
            dist.init_process_group(backend=self.config.hardware.backend)
            self.config.hardware.rank = dist.get_rank()
            self.config.hardware.world_size = dist.get_world_size()
            
            # Envolver modelo en DDP
            self.model = DDP(
                self.model,
                device_ids=[self.config.hardware.rank],
                output_device=self.config.hardware.rank
            )
            
            logger.info(f"Entrenamiento distribuido configurado (rank {self.config.hardware.rank})")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        n_targets = getattr(self.config.model, 'n_targets', 512)  # valor por defecto
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"[cyan]Entrenando (Epoch {epoch+1})", total=len(self.train_loader))
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image']
                labels = batch['labels']
                images = torch.stack(images).to(self.device)
                padded_labels = []
                for l in labels:
                    l = l.clone()
                    if l.numel() < n_targets:
                        l = torch.cat([l, torch.full((n_targets - l.numel(),), -1, dtype=l.dtype)])
                    padded_labels.append(l[:n_targets])
                targets = torch.stack(padded_labels).to(self.device).float()  # [B, n_targets]
                logits = self.model(images)  # [B, n_targets]
                mask_valid = (targets != -1)
                loss = calculate_loss(logits[mask_valid], targets[mask_valid], self.config.training.loss_function, self.config.training.class_weights)
                if self.config.hardware.use_amp:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.hardware.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.hardware.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                total_loss += loss.item()
                all_predictions.append(logits[mask_valid].detach().cpu())
                all_targets.append(targets[mask_valid].detach().cpu())
                progress.update(task, advance=1, description=f"[cyan]Entrenando (Epoch {epoch+1}) [Loss: {loss.item():.4f}]")
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(self.train_loader)
        return metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_images = []
        n_targets = getattr(self.config.model, 'n_targets', 512)
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"[magenta]Validando (Epoch {epoch+1})", total=len(self.val_loader))
                for batch_idx, batch in enumerate(self.val_loader):
                    images = batch['image']
                    labels = batch['labels']
                    images = torch.stack(images).to(self.device)
                    padded_labels = []
                    for l in labels:
                        l = l.clone()
                        if l.numel() < n_targets:
                            l = torch.cat([l, torch.full((n_targets - l.numel(),), -1, dtype=l.dtype)])
                        padded_labels.append(l[:n_targets])
                    targets = torch.stack(padded_labels).to(self.device).float()
                    logits = self.model(images)  # [B, n_targets]
                    mask_valid = (targets != -1)
                    loss = calculate_loss(logits[mask_valid], targets[mask_valid], self.config.training.loss_function, self.config.training.class_weights)
                    total_loss += loss.item()
                    all_predictions.append(logits[mask_valid].detach().cpu())
                    all_targets.append(targets[mask_valid].detach().cpu())
                    if batch_idx == 0:
                        all_images.append(images[:4].cpu())
                    progress.update(task, advance=1, description=f"[magenta]Validando (Epoch {epoch+1}) [Loss: {loss.item():.4f}]")
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        return metrics
    
    def update_scheduler(self, val_metric: float):
        """Actualizar scheduler basado en métrica de validación."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metric)
            else:
                self.scheduler.step()
    
    def check_early_stopping(self, val_metric: float) -> bool:
        """
        Verificar si se debe aplicar early stopping.
        
        Args:
            val_metric: Métrica de validación actual
            
        Returns:
            True si se debe parar el entrenamiento
        """
        if val_metric > self.best_metric + self.config.training.early_stopping_min_delta:
            self.best_metric = val_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            logger.info(f"Early stopping patience: {self.patience_counter}/{self.config.training.early_stopping_patience}")
            return self.patience_counter >= self.config.training.early_stopping_patience
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], is_best: bool = False):
        """Guardar checkpoint del modelo."""
        checkpoint_path = f"{self.config.logging.models_dir}/checkpoint_epoch_{epoch+1}.pth"
        
        # Asegurar que el directorio existe
        Path(self.config.logging.models_dir).mkdir(parents=True, exist_ok=True)
        
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            {**train_metrics, **val_metrics},
            checkpoint_path,
            is_best
        )
    
    def print_metrics_table(self, train_metrics, val_metrics, epoch):
        table = Table(title=f"Resumen de métricas - Epoch {epoch+1}", box=box.SIMPLE_HEAVY)
        table.add_column("Split", style="bold")
        table.add_column("Loss", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("IoU", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_row("[cyan]Train", f"{train_metrics['loss']:.4f}", f"{train_metrics['f1_score']:.4f}", f"{train_metrics['iou']:.4f}", f"{train_metrics['accuracy']:.4f}")
        table.add_row("[magenta]Val", f"{val_metrics['loss']:.4f}", f"{val_metrics['f1_score']:.4f}", f"{val_metrics['iou']:.4f}", f"{val_metrics['accuracy']:.4f}")
        console.print(table)

    def train(self, resume_from: Optional[str] = None):
        """
        Ejecutar entrenamiento completo.
        
        Args:
            resume_from: Ruta del checkpoint para resumir entrenamiento
        """
        logger.info("Iniciando entrenamiento...")
        
        # Configurar componentes
        self.setup_data()
        self.setup_model()
        
        if self.config.hardware.distributed:
            self.setup_distributed()
        
        # Resumir entrenamiento si se especifica
        start_epoch = 0
        if resume_from:
            start_epoch, _ = load_checkpoint(self.model, self.optimizer, resume_from)
            start_epoch += 1
            logger.info(f"Resumiendo entrenamiento desde época {start_epoch}")
        
        # Guardar configuración
        self.config.save(self.config.logging.output_dir)
        
        print()

        # Ciclo de entrenamiento
        best_f1 = 0.0
        best_epoch = 0
        for epoch in range(start_epoch, self.config.training.epochs):
            console.rule(f"[bold green]Época {epoch+1}/{self.config.training.epochs}")
            train_metrics = self.train_epoch(epoch)
            if epoch % self.config.training.val_frequency == 0 and self.val_loader is not None:
                val_metrics = self.validate_epoch(epoch)
                self.print_metrics_table(train_metrics, val_metrics, epoch)
                step = epoch * len(self.train_loader)
                log_metrics(
                    {**train_metrics, **val_metrics},
                    step,
                    use_wandb=self.config.logging.wandb,
                    use_tensorboard=self.config.logging.tensorboard,
                    tensorboard_writer=self.tensorboard_writer
                )
                for key in train_metrics:
                    if f'train_{key}' in self.metrics_history:
                        self.metrics_history[f'train_{key}'].append(train_metrics[key])
                for key in val_metrics:
                    if f'val_{key}' in self.metrics_history:
                        self.metrics_history[f'val_{key}'].append(val_metrics[key])
                # Guardar el mejor modelo SOLO EN MEMORIA
                if val_metrics['f1_score'] > best_f1:
                    best_f1 = val_metrics['f1_score']
                    best_epoch = epoch+1
                    console.print(f"[bold yellow]Nuevo mejor modelo (F1={best_f1:.4f}) guardado en memoria en epoch {best_epoch}")
                    self.best_model_state = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in self.model.state_dict().items()}
                    self.best_optimizer_state = self.optimizer.state_dict()
                    self.best_epoch = epoch
                    self.best_train_metrics = train_metrics.copy()
                    self.best_val_metrics = val_metrics.copy()
                self.update_scheduler(val_metrics['f1_score'])
                if self.check_early_stopping(val_metrics['f1_score']):
                    console.print(f"[bold red]Early stopping activado en epoch {epoch+1}")
                    break
                # Guardar checkpoint regular si corresponde
            else:
                step = epoch * len(self.train_loader)
                log_metrics(
                    train_metrics,
                    step,
                    prefix="train_",
                    use_wandb=self.config.logging.wandb,
                    use_tensorboard=self.config.logging.tensorboard,
                    tensorboard_writer=self.tensorboard_writer
                )
                console.print(f"[cyan]Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_score']:.4f}, IoU: {train_metrics['iou']:.4f}")
        
        # Guardar el mejor modelo a disco al final
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.optimizer.load_state_dict(self.best_optimizer_state)
            self.save_checkpoint(
                self.best_epoch,
                self.best_train_metrics,
                self.best_val_metrics,
                is_best=True
            )
            console.print(f"[bold green]Mejor modelo guardado en disco en epoch {self.best_epoch+1}")
        # Guardar modelo final si corresponde
        if self.config.training.save_last:
            self.save_checkpoint(
                self.config.training.epochs - 1,
                train_metrics,
                val_metrics if 'val_metrics' in locals() else train_metrics
            )
        
        # Graficar historial de métricas
        # (Eliminado: no se genera gráfico de historial de métricas)
        console.rule("[bold green]Entrenamiento completado")
        console.print(f"[bold green]Mejor F1: {best_f1:.4f} en epoch {best_epoch}")
        
        # Cerrar TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()


def main():
    """Función principal para ejecutar el entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de detección de daños")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML",default="config.yaml")
    parser.add_argument("--resume", type=str, help="Ruta al checkpoint para resumir")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--epochs", type=int, help="Número de épocas",default=1)
    parser.add_argument("--batch_size", type=int, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--model", type=str, help="Nombre del modelo",default='linear_per_tile')
    parser.add_argument("--distributed", action="store_true", help="Usar entrenamiento distribuido")
    parser.add_argument("--prueba", action="store_true", help="Si se usa, emplea dataloader_small en vez de full",default=True)
    args = parser.parse_args()
    
    # Cargar configuración
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Sobrescribir con argumentos de línea de comandos
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.model is not None:
        config.model.model_name = args.model
    if args.distributed:
        config.hardware.distributed = True
    # Nuevo: modo prueba
    config.data.split = 'small' if args.prueba else 'full'
    
    # Crear trainer y ejecutar entrenamiento
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main() 