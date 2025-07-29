"""
Módulo de entrenamiento para el proyecto AI for Earthquake Response.

Este módulo implementa el ciclo completo de entrenamiento y validación
para modelos de detección de daños por terremotos.

Adaptado para trabajar con el nuevo data_loader.py que maneja datos pre/post evento.
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
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from loguru import logger
import wandb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

console = Console()

from config import Config, get_default_config
from data_loader import EarthquakeTileDataset, SubsetEarthquakeTileDataset, custom_collate
from model import get_model
from model import SiameseCNNClassifier
from utils import (
    set_seed, setup_logging, calculate_metrics, calculate_loss,
    save_checkpoint, load_checkpoint, log_metrics, setup_wandb,
    visualize_predictions, plot_metrics_history
)
# Usar get_dataloaders_aug para entrenamiento con augmentation
from data_loader import get_dataloaders_aug


def get_dataloaders(config):
    """
    Crea o carga DataLoaders usando el nuevo EarthquakeTileDataset.
    """
    import torch
    import os
    # Permitir cargar DataLoader serializados (PyTorch >=2.6)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([torch.utils.data.dataloader.DataLoader])
    
    # Construir sufijo igual que en data_loader.py
    data_root = getattr(config.data, 'data_root', 'preprocessed')
    split = getattr(config.data, 'split', 'full')
    mode = getattr(config.data, 'mode', 'post_only')
    suffix = f"{split}_{mode}"
    if getattr(config.data, 'small_dataset', False):
        small_percentage = getattr(config.data, 'small_percentage', 0.2)
        suffix += f"_small{small_percentage:.0%}"
    
    dataloader_dir = "outputs/dataloaders"
    train_path = os.path.join(dataloader_dir, f"dataloader_train_{suffix}.pt")
    val_path = os.path.join(dataloader_dir, f"dataloader_val_{suffix}.pt")
    test_path = os.path.join(dataloader_dir, f"dataloader_test_{suffix}.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[INFO] Cargando dataloaders desde {dataloader_dir} (sufijo: {suffix})")
        train_loader = torch.load(train_path, map_location='cpu')
        val_loader = torch.load(val_path, map_location='cpu')
        test_loader = torch.load(test_path, map_location='cpu') if os.path.exists(test_path) else None
        print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}" + (f" | Test batches: {len(test_loader)}" if test_loader else ""))
        return train_loader, val_loader, test_loader
    else:
        print(f"[INFO] Dataloaders no encontrados en disco, creándolos desde cero... (sufijo: {suffix})")
        
        # Parámetros del dataset
        filter_has_data = getattr(config.data, 'filter_has_data', True)
        max_channels = getattr(config.data, 'max_channels', None)
        batch_size = config.data.batch_size
        
        # Optimización: Aumentar num_workers y usar pin_memory
        num_workers = min(os.cpu_count(), 8)  # Usar hasta 8 workers o el número de CPUs disponible
        pin_memory = True
        persistent_workers = True  # Mantener workers vivos entre épocas
        prefetch_factor = 2  # Prefetch de datos

        # Cargar dataset completo de forma más eficiente
        print("[INFO] Cargando dataset completo...")
        dataset_full = EarthquakeTileDataset(
            data_root=data_root,
            split=split,
            mode=mode,
            filter_has_data=filter_has_data,
            max_channels=max_channels
        )
        
        # Crear splits train/val de forma más eficiente
        print("[INFO] Creando splits train/val...")
        import numpy as np
        np.random.seed(config.seed)
        indices = np.arange(len(dataset_full))
        np.random.shuffle(indices)
        split_ratio = getattr(config.data, 'split_ratio', 0.8)
        n_train = int(len(indices) * split_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Crear metadatos para cada split
        meta_train = dataset_full.meta_df.iloc[train_indices].reset_index(drop=True)
        meta_val = dataset_full.meta_df.iloc[val_indices].reset_index(drop=True)
        
        print(f"[INFO] Split creado: Train={len(meta_train)} | Val={len(meta_val)}")
        
        # Crear datasets para train y val
        print("[INFO] Creando datasets de train y val...")
        dataset_train = SubsetEarthquakeTileDataset(
            data_root=data_root,
            split=split,
            mode=mode,
            filter_has_data=filter_has_data,
            max_channels=max_channels,
            meta_subset=meta_train
        )
        dataset_val = SubsetEarthquakeTileDataset(
            data_root=data_root,
            split=split,
            mode=mode,
            filter_has_data=filter_has_data,
            max_channels=max_channels,
            meta_subset=meta_val
        )
        
        # Crear dataloaders optimizados
        print("[INFO] Creando dataloaders...")
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate
        )
        
        print(f"[INFO] Datasets cargados: Train: {len(dataset_train)} tiles | Val: {len(dataset_val)} tiles")
        print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        
        # Guardar dataloaders para uso futuro (opcional)
        try:
            os.makedirs(dataloader_dir, exist_ok=True)
            print(f"[INFO] Guardando dataloaders en {dataloader_dir}...")
            torch.save(train_loader, train_path)
            torch.save(val_loader, val_path)
            print("[INFO] Dataloaders guardados exitosamente")
        except Exception as e:
            print(f"[WARNING] No se pudieron guardar los dataloaders: {e}")
        
        return train_loader, val_loader, None

class Trainer:
    """
    Clase principal para el entrenamiento de modelos.
    
    Maneja el ciclo completo de entrenamiento, validación y logging.
    Adaptado para el nuevo sistema de datos pre/post evento.
    """
    
    def __init__(self, config: Config, use_aug: bool = False):
        """
        Inicializar el trainer.
        
        Args:
            config: Configuración del experimento
        """
        self.config = config
        self.use_aug = use_aug
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
        
        # Variables para guardar el mejor modelo en memoria
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_epoch = 0
        self.best_train_metrics = None
        self.best_val_metrics = None
        # --- Opción 6: threshold óptimo por época ---
        self.best_threshold = 0.5
        # --- Opción 7: SWA ---
        self.swa_model = None
        self.swa_start_epoch = getattr(config.training, 'swa_start_epoch', None)
        self.swa_active = False
        logger.info("Trainer inicializado")
    
    def setup_data(self):
        """Cargar DataLoaders con o sin augmentation según el flag use_aug."""
        if self.use_aug:
            print("[INFO] Usando data augmentation y muestreo balanceado (get_dataloaders_aug)")
            self.train_loader, self.val_loader, self.test_loader = get_dataloaders_aug(self.config)
        else:
            print("[INFO] Entrenando SIN data augmentation (get_dataloaders clásico)")
            from data_loader import get_dataloaders
            self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.config)
        
        # Calcular class_weights de forma eficiente usando el dataset directamente
        print("[INFO] Calculando class weights...")
        if hasattr(self.train_loader.dataset, 'meta_df'):
            # Usar metadata del dataset si está disponible
            meta_df = self.train_loader.dataset.meta_df
            if 'damaged' in meta_df.columns:
                damaged_count = meta_df['damaged'].sum()
                total_count = len(meta_df)
                if damaged_count > 0 and total_count > damaged_count:
                    w0 = float(total_count / (2 * (total_count - damaged_count)))
                    w1 = float(total_count / (2 * damaged_count))
                    self.config.training.class_weights = [w0, w1]
                    print(f"[INFO] Class weights calculados: [no_damaged={w0:.3f}, damaged={w1:.3f}]")
                    print(f"[INFO] Distribución: {total_count-damaged_count} no dañados, {damaged_count} dañados")
                else:
                    self.config.training.class_weights = None
                    print("[INFO] No se calcularon class weights (dataset desbalanceado o sin daños)")
            else:
                self.config.training.class_weights = None
                print("[INFO] Columna 'damaged' no encontrada en metadata, usando class_weights=None")
        else:
            # Fallback: usar un muestreo pequeño para estimar
            print("[INFO] Metadata no disponible, estimando class weights con muestreo...")
            total = 0
            damaged = 0
            sample_size = min(100, len(self.train_loader))  # Muestrear máximo 100 batches
            
            for i, batch in enumerate(self.train_loader):
                if i >= sample_size:
                    break
                damaged += batch['damaged'].sum().item()
                total += batch['damaged'].numel()
            
            if damaged > 0 and total > damaged:
                # Estimar proporción total basada en la muestra
                damage_ratio = damaged / total
                w0 = float(1.0 / (2 * (1 - damage_ratio)))
                w1 = float(1.0 / (2 * damage_ratio))
                self.config.training.class_weights = [w0, w1]
                print(f"[INFO] Class weights estimados: [no_damaged={w0:.3f}, damaged={w1:.3f}]")
                print(f"[INFO] Muestra: {total-damaged} no dañados, {damaged} dañados (ratio: {damage_ratio:.3f})")
            else:
                self.config.training.class_weights = None
                print("[INFO] No se calcularon class weights (muestra desbalanceada)")
    
    def setup_model(self):
        """Configurar modelo, optimizador y scheduler."""
        logger.info("Configurando modelo...")
        
        # Crear modelo
        backbone_weights_path = getattr(self.config, 'backbone_weights_path', None)
        self.model = get_model(self.config.model, backbone_weights_path=backbone_weights_path)
        self.model = self.model.to(self.device)
        
        # Congelar backbone si se cargan pesos preentrenados y freeze_backbone es True
        freeze_backbone = getattr(self.config, 'freeze_backbone', True)
        if backbone_weights_path is not None and hasattr(self.model, 'backbone') and freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] Backbone congelado (no se entrena) por tener pesos preentrenados: {backbone_weights_path}")
        
        # Configurar optimizador
        if self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Optimizador {self.config.training.optimizer} no soportado")
        
        # --- Scheduler con warm-up + CosineAnnealingWarmRestarts ---
        if self.config.training.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
            warmup_epochs = getattr(self.config.training, 'warmup_epochs', 5)
            t_0 = getattr(self.config.training, 'scheduler_t0', 10)
            total_epochs = self.config.training.epochs
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                return 1.0
            self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=t_0)
            self.use_warmup = True
            self._current_epoch = 0
        elif self.config.training.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
            self.use_warmup = False
        elif self.config.training.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                verbose=True
            )
            self.use_warmup = False
        else:
            self.use_warmup = False
        
        # Configurar mixed precision
        if self.config.hardware.use_amp:
            self.scaler = GradScaler(device='cuda')
        
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
        """Entrenar una época con optimizaciones de rendimiento."""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        all_predictions = []
        all_targets = []
        threshold = getattr(self.config.training, 'threshold', 0.5)
        label_smoothing = getattr(self.config.training, 'label_smoothing', 0.0)
        n_batches = len(self.train_loader)
        
        # Configurar barra de progreso con información detallada
        pbar = tqdm(
            total=n_batches, 
            desc=f"Epoch {epoch+1} [Train]", 
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        scaler = self.scaler if self.config.hardware.use_amp else None
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            building_mask = batch['building_mask'].to(self.device, non_blocking=True) if 'building_mask' in batch else None
            targets = batch['damaged'].to(self.device, non_blocking=True)
            
            if scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    if isinstance(self.model, SiameseCNNClassifier):
                        logits = self.model(images, building_mask)
                    else:
                        logits = self.model(images)
                    loss = calculate_loss(
                        logits, targets,
                        self.config.training.loss_function,
                        self.config.training.class_weights,
                        label_smoothing=label_smoothing
                    )
                scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.hardware.max_grad_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                if isinstance(self.model, SiameseCNNClassifier):
                    logits = self.model(images, building_mask)
                else:
                    logits = self.model(images)
                loss = calculate_loss(
                    logits, targets,
                    self.config.training.loss_function,
                    self.config.training.class_weights,
                    label_smoothing=label_smoothing
                )
                loss.backward()
                if (batch_idx + 1) % self.config.hardware.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.hardware.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.detach()
            
            # Calcular métricas en tiempo real
            with torch.no_grad():
                if logits.ndim == 2 and logits.shape[1] == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(logits)
                
                # Accuracy en tiempo real
                preds = (probs > threshold).float()
                correct = (preds == targets).sum().item()
                running_correct += correct
                running_total += targets.numel()
                running_loss += loss.item()
                
                all_predictions.append(probs)
                all_targets.append(targets)
            
            # Actualizar barra de progreso con métricas
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = running_correct / running_total if running_total > 0 else 0.0
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{avg_acc:.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            pbar.update(1)
        
        pbar.close()
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets, threshold=threshold)
        metrics['loss'] = (total_loss / n_batches).item()
        return metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validar una época con optimizaciones de rendimiento."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        all_predictions = []
        all_targets = []
        threshold = getattr(self.config.training, 'threshold', 0.5)
        label_smoothing = getattr(self.config.training, 'label_smoothing', 0.0)
        n_batches = len(self.val_loader)
        
        # Configurar barra de progreso con información detallada
        pbar = tqdm(
            total=n_batches, 
            desc=f"Epoch {epoch+1} [Val]", 
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.hardware.use_amp):
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch['image'].to(self.device, non_blocking=True)
                building_mask = batch['building_mask'].to(self.device, non_blocking=True) if 'building_mask' in batch else None
                targets = batch['damaged'].to(self.device, non_blocking=True)
                
                if isinstance(self.model, SiameseCNNClassifier):
                    logits = self.model(images, building_mask)
                else:
                    logits = self.model(images)
                
                loss = calculate_loss(
                    logits, targets,
                    self.config.training.loss_function,
                    self.config.training.class_weights,
                    label_smoothing=label_smoothing
                )
                
                total_loss += loss
                
                # Calcular métricas en tiempo real
                if logits.ndim == 2 and logits.shape[1] == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(logits)
                
                # Accuracy en tiempo real
                preds = (probs > threshold).float()
                correct = (preds == targets).sum().item()
                running_correct += correct
                running_total += targets.numel()
                running_loss += loss.item()
                
                all_predictions.append(probs)
                all_targets.append(targets)
                
                # Actualizar barra de progreso con métricas
                avg_loss = running_loss / (batch_idx + 1)
                avg_acc = running_correct / running_total if running_total > 0 else 0.0
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{avg_acc:.3f}'
                })
                pbar.update(1)
        
        pbar.close()
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets, threshold=threshold)
        metrics['loss'] = (total_loss / n_batches).item()
        return metrics
    
    def update_scheduler(self, val_metric: float):
        """Actualizar scheduler basado en métrica de validación."""
        if hasattr(self, 'use_warmup') and self.use_warmup:
            # Warmup las primeras warmup_epochs
            if self._current_epoch < getattr(self.config.training, 'warmup_epochs', 5):
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(self._current_epoch - getattr(self.config.training, 'warmup_epochs', 5))
            self._current_epoch += 1
            # Loggear LR en TensorBoard si está activo
            if self.tensorboard_writer:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.tensorboard_writer.add_scalar(f'LR/group_{i}', param_group['lr'], self._current_epoch)
        elif self.scheduler is not None:
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
        """Imprimir tabla de métricas con información detallada."""
        table = Table(title=f"Resumen de métricas - Epoch {epoch+1}", box=box.SIMPLE_HEAVY)
        table.add_column("Split", style="bold")
        table.add_column("Loss", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("IoU", justify="right")
        
        # Obtener métricas adicionales si están disponibles
        train_precision = train_metrics.get('precision', 0.0)
        train_recall = train_metrics.get('recall', 0.0)
        val_precision = val_metrics.get('precision', 0.0)
        val_recall = val_metrics.get('recall', 0.0)
        
        table.add_row(
            "[cyan]Train", 
            f"{train_metrics['loss']:.4f}", 
            f"{train_metrics['f1_score']:.4f}", 
            f"{train_precision:.4f}",
            f"{train_recall:.4f}",
            f"{train_metrics['accuracy']:.4f}", 
            f"{train_metrics['iou']:.4f}"
        )
        table.add_row(
            "[magenta]Val", 
            f"{val_metrics['loss']:.4f}", 
            f"{val_metrics['f1_score']:.4f}", 
            f"{val_precision:.4f}",
            f"{val_recall:.4f}",
            f"{val_metrics['accuracy']:.4f}", 
            f"{val_metrics['iou']:.4f}"
        )
        console.print(table)
        
        # Información adicional sobre el dataset (optimizada)
        if hasattr(self, 'train_loader') and hasattr(self, 'val_loader'):
            # Calcular distribución de clases de forma eficiente usando metadata
            train_damaged = 0
            train_total = 0
            val_damaged = 0
            val_total = 0
            
            # Usar metadata si está disponible (más rápido)
            if hasattr(self.train_loader.dataset, 'meta_df'):
                train_meta = self.train_loader.dataset.meta_df
                if 'damaged' in train_meta.columns:
                    train_damaged = train_meta['damaged'].sum()
                    train_total = len(train_meta)
                
                val_meta = self.val_loader.dataset.meta_df
                if 'damaged' in val_meta.columns:
                    val_damaged = val_meta['damaged'].sum()
                    val_total = len(val_meta)
            else:
                # Fallback: muestreo pequeño para estimar
                sample_size = min(10, len(self.train_loader))
                for i, batch in enumerate(self.train_loader):
                    if i >= sample_size:
                        break
                    train_damaged += batch['damaged'].sum().item()
                    train_total += batch['damaged'].size(0)
                
                # Estimar proporción total
                if train_total > 0:
                    damage_ratio = train_damaged / train_total
                    train_total = len(self.train_loader.dataset)
                    train_damaged = int(train_total * damage_ratio)
                
                # Mismo para val
                sample_size = min(10, len(self.val_loader))
                for i, batch in enumerate(self.val_loader):
                    if i >= sample_size:
                        break
                    val_damaged += batch['damaged'].sum().item()
                    val_total += batch['damaged'].size(0)
                
                if val_total > 0:
                    damage_ratio = val_damaged / val_total
                    val_total = len(self.val_loader.dataset)
                    val_damaged = int(val_total * damage_ratio)
            
            if train_total > 0 and val_total > 0:
                console.print(f"[bold blue]Distribución de clases:")
                console.print(f"  Train: {train_damaged}/{train_total} dañados ({train_damaged/train_total*100:.1f}%)")
                console.print(f"  Val: {val_damaged}/{val_total} dañados ({val_damaged/val_total*100:.1f}%)")
            
            # Explicación del comportamiento de métricas
            if val_metrics['f1_score'] == 0.0 and val_metrics['accuracy'] > 0.5:
                console.print(f"[bold yellow]⚠️  Análisis de métricas:")
                console.print(f"  - F1=0.0 pero Accuracy={val_metrics['accuracy']:.4f}")
                console.print(f"  - Esto indica que el modelo está prediciendo principalmente la clase mayoritaria")
                console.print(f"  - En datasets desbalanceados, F1 es más confiable que Accuracy")
                console.print(f"  - Considera usar class_weights o focal_loss para mejorar F1")
            
            # Mostrar métricas detalladas si están disponibles
            if 'tp' in val_metrics and 'tn' in val_metrics and 'fp' in val_metrics and 'fn' in val_metrics:
                console.print(f"[bold green]Matriz de confusión (Val):")
                console.print(f"  TP: {val_metrics['tp']}, TN: {val_metrics['tn']}")
                console.print(f"  FP: {val_metrics['fp']}, FN: {val_metrics['fn']}")
                console.print(f"  Precision: {val_metrics.get('precision', 0):.4f} (TP/(TP+FP))")
                console.print(f"  Recall: {val_metrics.get('recall', 0):.4f} (TP/(TP+FN))")

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
        
        # --- Mostrar resumen de configuración ---
        console.rule("[bold blue]Configuración del Entrenamiento[/bold blue]")
        console.print("[bold cyan]Parámetros del modelo:[/bold cyan]")
        console.print(f"  Modelo: [yellow]{self.config.model.model_name}[/yellow]")
        console.print(f"  Learning rate: [yellow]{self.config.training.learning_rate}[/yellow]")
        console.print(f"  Optimizador: [yellow]{self.config.training.optimizer}[/yellow]")
        console.print(f"  Scheduler: [yellow]{self.config.training.scheduler}[/yellow]")
        if hasattr(self.config.training, 'warmup_epochs'):
            console.print(f"  Warmup epochs: [yellow]{self.config.training.warmup_epochs}[/yellow]")
        console.print(f"  Épocas totales: [yellow]{self.config.training.epochs}[/yellow]")
        console.print("\n[bold cyan]Parámetros de pérdida y métricas:[/bold cyan]")
        console.print(f"  Función de pérdida: [yellow]{self.config.training.loss_function}[/yellow]")
        console.print(f"  Label smoothing: [yellow]{getattr(self.config.training, 'label_smoothing', 0.0)}[/yellow]")
        console.print(f"  Class weights: [yellow]{self.config.training.class_weights if hasattr(self.config.training, 'class_weights') else 'No'}[/yellow]")
        console.print("\n[bold cyan]Parámetros de datos:[/bold cyan]")
        console.print(f"  Batch size: [yellow]{self.config.data.batch_size}[/yellow]")
        console.print(f"  Data augmentation: [yellow]{'Sí' if self.use_aug else 'No'}[/yellow]")
        console.print(f"  Modo de datos: [yellow]{self.config.data.mode}[/yellow]")
        console.print(f"  Canales máximos: [yellow]{self.config.data.max_channels}[/yellow]")
        if hasattr(self.config.training, 'swa_start_epoch') and self.config.training.swa_start_epoch is not None:
            console.print("\n[bold cyan]Stochastic Weight Averaging:[/bold cyan]")
            console.print(f"  Inicio en época: [yellow]{self.config.training.swa_start_epoch}[/yellow]")
        console.rule()
        print()  # Línea en blanco para separar
        
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
        best_f1 = self.best_metric if hasattr(self, 'best_metric') else 0.0
        best_epoch = 0
        # --- Opción 7: Inicializar SWA si corresponde ---
        if self.swa_start_epoch is not None:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.config.training.learning_rate * 0.1)
        
        # Barra de progreso general para épocas
        epoch_pbar = tqdm(
            range(start_epoch, self.config.training.epochs),
            desc="Training Progress",
            unit="epoch",
            position=0,
            leave=True
        )
        
        for epoch in epoch_pbar:
            console.rule(f"[bold green]Época {epoch+1}/{self.config.training.epochs}")
            
            train_metrics = self.train_epoch(epoch)
            
            # --- Protección para val_loader vacío ---
            if epoch % self.config.training.val_frequency == 0 and self.val_loader is not None and len(self.val_loader) > 0:
                self.model.eval()  # ➜ desactiva Dropout y pone BN en modo eval
                thresholds = np.linspace(0.05, 0.95, 19)
                tps = np.zeros_like(thresholds)
                fps = np.zeros_like(thresholds)
                fns = np.zeros_like(thresholds)
                total_loss_val = 0.0
                n_val = 0
                all_predictions = []
                all_tgts = []
                for batch in self.val_loader:
                    images = batch['image'].to(self.device)
                    building_mask = batch['building_mask'].to(self.device) if 'building_mask' in batch else None
                    targets = batch['damaged'].to(self.device)
                    if isinstance(self.model, SiameseCNNClassifier):
                        logits = self.model(images, building_mask)
                    else:
                        logits = self.model(images)
                    if logits.ndim == 2 and logits.shape[1] == 2:
                        probs = torch.softmax(logits, dim=1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits)
                    all_predictions.append(logits.detach().cpu())  # Guardar logits, no probs
                    all_tgts.append(targets.cpu())
                    for i, t in enumerate(thresholds):
                        preds_bin = (probs > t).float()
                        tps[i] += ((preds_bin == 1) & (targets == 1)).sum().item()
                        fps[i] += ((preds_bin == 1) & (targets == 0)).sum().item()
                        fns[i] += ((preds_bin == 0) & (targets == 1)).sum().item()
                    total_loss_val += calculate_loss(logits, targets, self.config.training.loss_function, self.config.training.class_weights, label_smoothing=getattr(self.config.training, 'label_smoothing', 0.0)).item()
                    n_val += 1
                best_f1_epoch = 0.0
                best_t = 0.5
                for i, t in enumerate(thresholds):
                    precision = tps[i] / (tps[i] + fps[i] + 1e-7)
                    recall = tps[i] / (tps[i] + fns[i] + 1e-7)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
                    if f1 > best_f1_epoch:
                        best_f1_epoch = f1
                        best_t = t
                self.best_threshold = best_t
                console.print(f"[bold cyan]Threshold óptimo en val: {best_t:.2f} (F1={best_f1_epoch:.4f})[/bold cyan]")
                # Calcular métricas finales con el threshold óptimo usando los arrays guardados
                if all_predictions and all_tgts:
                    all_predictions_cat = torch.cat(all_predictions, dim=0)
                    all_targets_cat = torch.cat(all_tgts, dim=0)
                    val_metrics = calculate_metrics(all_predictions_cat, all_targets_cat, threshold=best_t)
                    val_metrics['loss'] = total_loss_val / max(1, n_val)
                else:
                    val_metrics = {k: train_metrics.get(k, 0.0) for k in train_metrics}
                self.print_metrics_table(train_metrics, val_metrics, epoch)
                
                step = epoch * len(self.train_loader)
                log_metrics(
                    {**train_metrics, **val_metrics, 'best_threshold': self.best_threshold},
                    step,
                    use_wandb=self.config.logging.wandb,
                    use_tensorboard=self.config.logging.tensorboard,
                    tensorboard_writer=self.tensorboard_writer
                )
                
                # Actualizar historial
                for key in train_metrics:
                    if f'train_{key}' in self.metrics_history:
                        self.metrics_history[f'train_{key}'].append(train_metrics[key])
                for key in val_metrics:
                    if f'val_{key}' in self.metrics_history:
                        self.metrics_history[f'val_{key}'].append(val_metrics[key])
                
                # Guardar el mejor modelo en disco INMEDIATAMENTE si mejora F1
                if val_metrics['f1_score'] > best_f1:
                    best_f1 = val_metrics['f1_score']
                    self.best_metric = best_f1
                    best_epoch = epoch+1
                    self.best_model_state = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in self.model.state_dict().items()}
                    self.best_optimizer_state = self.optimizer.state_dict()
                    self.best_epoch = epoch
                    self.best_train_metrics = train_metrics.copy()
                    self.best_val_metrics = val_metrics.copy()
                # --- Opción 7: Actualizar SWA si corresponde ---
                if self.swa_model is not None and epoch >= self.swa_start_epoch:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                    self.swa_active = True
                self.update_scheduler(val_metrics['f1_score'])
                
                # --- Opción 9: Early stopping robusto (reset si baja LR) ---
                if hasattr(self, 'scheduler') and hasattr(self.scheduler, 'get_last_lr'):
                    current_lr = self.scheduler.get_last_lr()[0]
                    if hasattr(self, '_last_lr') and current_lr < self._last_lr:
                        self.patience_counter = 0
                        console.print("[bold green]LR bajó, early stopping patience reseteada[/bold green]")
                    self._last_lr = current_lr
                # Early stopping: evaluar cada época usando la última métrica disponible
                if self.check_early_stopping(best_f1):
                    console.print(f"[bold red]Early stopping activado en epoch {epoch+1}")
                    break
            else:
                # Si no hay val_loader, asigna métricas dummy o copia las de train
                val_metrics = {k: train_metrics.get(k, 0.0) for k in train_metrics}
                self.print_metrics_table(train_metrics, val_metrics, epoch)
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
            
            # Actualizar barra de progreso general
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_metrics["loss"]:.4f}',
                'Train_F1': f'{train_metrics["f1_score"]:.4f}',
                'Best_F1': f'{best_f1:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        epoch_pbar.close()
        
        # Ya no se guarda el mejor modelo aquí, solo el "last" si corresponde
        
        # Guardar el mejor modelo al final del entrenamiento
        if self.best_model_state is not None:
            checkpoint_path = f"{self.config.logging.models_dir}/best.pth"
            Path(self.config.logging.models_dir).mkdir(parents=True, exist_ok=True)
            # Cargar el mejor estado en el modelo antes de guardar
            self.model.load_state_dict(self.best_model_state)
            self.optimizer.load_state_dict(self.best_optimizer_state)
            save_checkpoint(
                self.model,
                self.optimizer,
                self.best_epoch,
                {**self.best_train_metrics, **self.best_val_metrics},
                checkpoint_path,
                is_best=True
            )
        
        # Guardar el threshold óptimo en el config y en outputs/config.yaml
        import os
        os.makedirs('outputs', exist_ok=True)
        self.config.training.threshold = self.best_threshold
        self.config.save('outputs/config.yaml')
        
        console.rule("[bold green]Entrenamiento completado")
        console.print(f"[bold green]Mejor F1: {best_f1:.4f} en epoch {best_epoch}")
        
        # Cerrar TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

    # --- Opción 8: Test-time augmentation (TTA) ---
    def predict_tta(self, loader, n_aug=4):
        """
        Realiza Test-Time Augmentation (TTA) usando hasta n_aug vistas:
        - 1: original
        - 2: + flip horizontal
        - 3: + flip vertical
        - 4: + rotación 90º
        (puedes ampliar con más augmentaciones si lo deseas)
        """
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device)
                building_mask = batch['building_mask'].to(self.device) if 'building_mask' in batch else None
                aug_images = [images]
                if n_aug > 1:
                    aug_images.append(torch.flip(images, dims=[-1]))  # horizontal
                if n_aug > 2:
                    aug_images.append(torch.flip(images, dims=[-2]))  # vertical
                if n_aug > 3:
                    aug_images.append(torch.rot90(images, k=1, dims=[-2, -1]))
                probs = 0
                for aug in aug_images[:n_aug]:
                    if isinstance(self.model, SiameseCNNClassifier):
                        logits = self.model(aug, building_mask)
                    else:
                        logits = self.model(aug)
                    if logits.ndim == 2 and logits.shape[1] == 2:
                        probs += torch.softmax(logits, dim=1)[:, 1]
                    else:
                        probs += torch.sigmoid(logits)
                probs = probs / len(aug_images[:n_aug])
                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0)

# --- Opción 10: Leave-one-event-out CV (estructura) ---
def leave_one_event_out_cv(config):
    from data_loader import EarthquakeTileDataset
    meta = EarthquakeTileDataset(
        data_root=getattr(config.data, 'data_root', 'preprocessed'),
        split=getattr(config.data, 'split', 'full'),
        mode=getattr(config.data, 'mode', 'post_only'),
        filter_has_data=getattr(config.data, 'filter_has_data', True),
        max_channels=getattr(config.data, 'max_channels', 3)
    ).meta_df
    unique_events = meta['event'].unique()
    for fold, event in enumerate(unique_events):
        print(f"[CV] Fold {fold+1}/{len(unique_events)}: evento {event}")
        # Aquí deberías crear splits train/val dejando fuera el evento actual
        # y entrenar un modelo por fold, guardando resultados
        # ...


def main():
    import shutil
    import os
    # Borrar la carpeta outputs/config.yaml si existe
    config_yaml_dir = os.path.join('outputs', 'config.yaml')
    if os.path.isdir(config_yaml_dir):
        print(f"[INFO] Borrando directorio conflictivo: {config_yaml_dir}")
        shutil.rmtree(config_yaml_dir, ignore_errors=True)
    """Función principal para ejecutar el entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de detección de daños")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML", default="config.yaml")
    parser.add_argument("--resume", type=str, help="Ruta al checkpoint para resumir")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--epochs", type=int, help="Número de épocas", default=1)
    parser.add_argument("--batch_size",default=16, type=int, help="Tamaño de batch")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--model", type=str, help="Nombre del modelo", default='siamese_cnn_classifier')
    # Cambiar optimizador y scheduler por defecto
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizador (adam, adamw, sgd)")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler (cosine, step, plateau)")
    parser.add_argument("--distributed", action="store_true", help="Usar entrenamiento distribuido")
    parser.add_argument("--small_dataset", action="store_true", help="Usar dataset small para pruebas rápidas")
    parser.add_argument("--small_percentage", type=float, default=0.2, help="Porcentaje para dataset small")
    parser.add_argument("--mode", type=str, default='pre_post', choices=['pre_post', 'pre_only', 'post_only'],
                       help='Modo de datos: pre_post, pre_only, post_only')
    parser.add_argument("--max_channels", type=int, help="Máximo número de canales a usar",default=3)
    # NUEVO: Argumentos para tipo de loss y threshold
    parser.add_argument("--loss_type", type=str, default="bce_dice", choices=["bce", "dice", "bce_dice", "focal"], help="Tipo de función de pérdida")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para binarización de clase positiva")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Épocas de warmup para el scheduler")
    parser.add_argument("--scheduler_t0", type=int, default=10, help="T_0 para CosineAnnealingWarmRestarts")
    parser.add_argument("--use_aug", action="store_true",default=True, help="Usar data augmentation y muestreo balanceado (get_dataloaders_aug)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing binario (0.0 = sin smoothing, típico 0.05)")
    parser.add_argument("--swa_start_epoch", type=int, default=None, help="Época a partir de la cual activar SWA (Stochastic Weight Averaging)")
    parser.add_argument("--backbone_weights_path", type=str, default="outputs/models/cnnbackbone_simclr.pth", help="Ruta a los pesos preentrenados del backbone (SimCLR)")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Congelar el backbone si se cargan pesos preentrenados (por defecto: True)")
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
    if args.optimizer:
        config.training.optimizer = args.optimizer
    if args.scheduler:
        config.training.scheduler = args.scheduler
    if args.distributed:
        config.hardware.distributed = True
    if args.small_dataset:
        config.data.split = 'small'
        config.data.small_percentage = args.small_percentage
    if args.mode:
        config.data.mode = args.mode
    if args.max_channels:
        config.data.max_channels = args.max_channels
    if args.loss_type:
        config.training.loss_function = args.loss_type
    if args.threshold is not None:
        config.training.threshold = args.threshold
    if args.warmup_epochs is not None:
        config.training.warmup_epochs = args.warmup_epochs
    if args.scheduler_t0 is not None:
        config.training.scheduler_t0 = args.scheduler_t0
    if args.label_smoothing is not None:
        config.training.label_smoothing = args.label_smoothing
    if args.swa_start_epoch is not None:
        config.training.swa_start_epoch = args.swa_start_epoch
    if args.backbone_weights_path is not None:
        config.backbone_weights_path = args.backbone_weights_path
    config.freeze_backbone = args.freeze_backbone
    
    # Crear trainer y ejecutar entrenamiento
    # --- Ajuste automático de in_channels para siamese_cnn_classifier ---
    if config.model.model_name == "siamese_cnn_classifier":
        # Si el modo es pre_post, el dataloader entrega 2*C canales
        if hasattr(config.data, 'mode') and config.data.mode == 'pre_post':
            # Si el usuario puso in_channels = C, lo duplicamos
            if getattr(config.model, 'in_channels', None) is not None:
                if config.model.in_channels * 2 != config.data.max_channels * 2:
                    # Forzamos in_channels a 2*C
                    config.model.in_channels = config.data.max_channels * 2
            else:
                config.model.in_channels = config.data.max_channels * 2
        else:
            # Si no es pre_post, advertir
            print("[WARNING] El modelo siamese_cnn_classifier está pensado para modo pre_post (input 2*C canales)")
        # Guardar max_channels en config.model para el backbone
        config.model.max_channels = config.data.max_channels
    trainer = Trainer(config, use_aug=args.use_aug)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main() 