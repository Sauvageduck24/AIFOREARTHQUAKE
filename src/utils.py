"""
Módulo de utilidades para el proyecto AI for Earthquake Response.

Este módulo contiene funciones auxiliares para métricas, logging,
visualización y otras tareas comunes.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from loguru import logger
import wandb
from torch.utils.tensorboard import SummaryWriter
import cv2


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Configurar semilla para reproducibilidad.
    
    Args:
        seed: Semilla a usar
        deterministic: Si usar modo determinístico
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    logger.info(f"Semilla configurada: {seed} (determinístico: {deterministic})")


def setup_logging(
    log_dir: str = "outputs/logs",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
):
    """
    Configurar logging con loguru.
    
    Args:
        log_dir: Directorio para logs
        level: Nivel de logging
        rotation: Rotación de archivos de log
        retention: Retención de archivos de log
    """
    # Crear directorio de logs
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configurar loguru
    logger.remove()  # Remover handler por defecto
    
    # Handler para consola
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Handler para archivo
    logger.add(
        f"{log_dir}/training.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    logger.info("Logging configurado")


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcular métricas de evaluación.
    
    Args:
        predictions: Predicciones del modelo (logits)
        targets: Targets reales
        threshold: Umbral para binarización
        
    Returns:
        Diccionario con métricas calculadas
    """
    # Si predictions es 1D, no accedemos a shape[1]
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    if targets.ndim > 1 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    
    # Binarizar predicciones
    preds_bin = (torch.sigmoid(predictions) > threshold).float()
    targets_bin = targets.float()
    
    # Calcular métricas básicas
    tp = ((preds_bin == 1) & (targets_bin == 1)).sum().item()
    tn = ((preds_bin == 0) & (targets_bin == 0)).sum().item()
    fp = ((preds_bin == 1) & (targets_bin == 0)).sum().item()
    fn = ((preds_bin == 0) & (targets_bin == 1)).sum().item()
    
    # Evitar división por cero
    epsilon = 1e-7
    
    # Métricas de clasificación
    accuracy = (preds_bin == targets_bin).float().mean().item()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + epsilon)
    
    # Dice Coefficient
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    
    # Calcular AUC-ROC
    try:
        fpr, tpr, _ = roc_curve(targets_bin.cpu().numpy(), preds_bin.cpu().numpy())
        auc_roc = auc(fpr, tpr)
    except:
        auc_roc = 0.0
    
    # Calcular AUC-PR
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(targets_bin.cpu().numpy(), preds_bin.cpu().numpy())
        auc_pr = auc(recall_curve, precision_curve)
    except:
        auc_pr = 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'iou': float(iou),
        'dice': float(dice),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def calculate_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "bce_dice",
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calcular función de pérdida.
    
    Args:
        predictions: Predicciones del modelo
        targets: Targets reales
        loss_type: Tipo de pérdida
        class_weights: Pesos de clases (opcional)
        
    Returns:
        Tensor con el valor de pérdida
    """
    if loss_type == "bce":
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        if targets.ndim > 1 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=predictions.device, dtype=predictions.dtype)
            loss = F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=class_weights)
        else:
            loss = F.binary_cross_entropy_with_logits(predictions, targets)
    
    elif loss_type == "dice":
        loss = dice_loss(predictions, targets)
    
    elif loss_type == "bce_dice":
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=class_weights)
        dice_loss_val = dice_loss(predictions, targets)
        loss = 0.5 * bce_loss + 0.5 * dice_loss_val
    
    elif loss_type == "focal":
        loss = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)
    
    else:
        raise ValueError(f"Tipo de pérdida {loss_type} no soportado")
    
    return loss


def dice_loss(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calcular pérdida Dice.
    
    Args:
        predictions: Predicciones del modelo
        targets: Targets reales
        smooth: Factor de suavizado
        
    Returns:
        Tensor con el valor de pérdida Dice
    """
    if predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions)
    else:
        probs = F.softmax(predictions, dim=1)
        probs = probs[:, 1:2]
    
    # Aplanar tensores
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    # Calcular Dice
    intersection = (probs_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    
    return 1 - dice


def focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Calcular pérdida Focal.
    
    Args:
        predictions: Predicciones del modelo
        targets: Targets reales
        alpha: Parámetro alpha
        gamma: Parámetro gamma
        
    Returns:
        Tensor con el valor de pérdida Focal
    """
    if predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions)
    else:
        probs = F.softmax(predictions, dim=1)
        probs = probs[:, 1:2]
    
    # Aplanar tensores
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    # Calcular Focal Loss
    bce_loss = F.binary_cross_entropy(probs_flat, targets_flat, reduction='none')
    pt = torch.where(targets_flat == 1, probs_flat, 1 - probs_flat)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    
    return focal_loss.mean()


def visualize_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    save_path: Optional[str] = None,
    num_samples: int = 4
) -> None:
    """
    Visualizar predicciones del modelo.
    
    Args:
        images: Imágenes de entrada
        targets: Targets reales
        predictions: Predicciones del modelo
        save_path: Ruta para guardar visualización
        num_samples: Número de muestras a visualizar
    """
    # Limitar número de muestras
    num_samples = min(num_samples, images.shape[0])
    
    # Aplicar sigmoid a predicciones
    if predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions)
    else:
        probs = F.softmax(predictions, dim=1)
        probs = probs[:, 1:2]
    
    # Convertir a numpy
    images_np = images[:num_samples].cpu().numpy()
    targets_np = targets[:num_samples].cpu().numpy()
    probs_np = probs[:num_samples].cpu().numpy()
    
    # Crear figura
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Imagen pre-terremoto (primeros 3 canales)
        axes[i, 0].imshow(np.transpose(images_np[i, :3], (1, 2, 0)))
        axes[i, 0].set_title('Pre-Earthquake')
        axes[i, 0].axis('off')
        
        # Imagen post-terremoto (últimos 3 canales)
        axes[i, 1].imshow(np.transpose(images_np[i, 3:], (1, 2, 0)))
        axes[i, 1].set_title('Post-Earthquake')
        axes[i, 1].axis('off')
        
        # Target
        axes[i, 2].imshow(targets_np[i, 0], cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Predicción
        axes[i, 3].imshow(probs_np[i, 0], cmap='hot', vmin=0, vmax=1)
        axes[i, 3].set_title('Prediction')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualización guardada en {save_path}")
    
    plt.show()


def plot_metrics_history(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Graficar historial de métricas.
    
    Args:
        metrics_history: Diccionario con historial de métricas
        save_path: Ruta para guardar gráfico
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['loss', 'accuracy', 'f1_score', 'iou']
    titles = ['Loss', 'Accuracy', 'F1-Score', 'IoU']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        if metric in metrics_history:
            train_metric = metrics_history.get(f'train_{metric}', [])
            val_metric = metrics_history.get(f'val_{metric}', [])
            
            epochs = range(1, len(train_metric) + 1)
            
            axes[i].plot(epochs, train_metric, 'b-', label=f'Train {title}')
            if val_metric:
                axes[i].plot(epochs, val_metric, 'r-', label=f'Val {title}')
            
            axes[i].set_title(title)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(title)
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de métricas guardado en {save_path}")
    
    plt.show()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Guardar checkpoint del modelo.
    
    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Época actual
        metrics: Métricas actuales
        save_path: Ruta para guardar checkpoint
        is_best: Si es el mejor modelo hasta ahora
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'is_best': is_best
    }
    
    # Guardar checkpoint
    torch.save(checkpoint, save_path)
    
    # Guardar mejor modelo
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"Mejor modelo guardado en {best_path}")
    
    logger.info(f"Checkpoint guardado en {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> Tuple[int, Dict[str, float]]:
    """
    Cargar checkpoint del modelo.
    
    Args:
        model: Modelo a cargar
        optimizer: Optimizador
        checkpoint_path: Ruta del checkpoint
        
    Returns:
        Tupla con época y métricas
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    logger.info(f"Checkpoint cargado desde {checkpoint_path} (época {epoch})")
    
    return epoch, metrics


def setup_wandb(
    project_name: str,
    entity: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    Configurar Weights & Biases.
    
    Args:
        project_name: Nombre del proyecto
        entity: Entidad de W&B
        config: Configuración a loggear
        tags: Tags para el experimento
    """
    try:
        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            tags=tags
        )
        logger.info("Weights & Biases configurado")
    except Exception as e:
        logger.warning(f"No se pudo configurar W&B: {e}")


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    use_wandb: bool = True,
    use_tensorboard: bool = True,
    tensorboard_writer: Optional[SummaryWriter] = None
) -> None:
    """
    Loggear métricas en W&B y TensorBoard.
    
    Args:
        metrics: Métricas a loggear
        step: Paso actual
        prefix: Prefijo para las métricas
        use_wandb: Si usar W&B
        use_tensorboard: Si usar TensorBoard
        tensorboard_writer: Writer de TensorBoard
    """
    # Loggear en W&B
    if use_wandb and wandb.run is not None:
        wandb_metrics = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        wandb.log(wandb_metrics, step=step)
    
    # Loggear en TensorBoard
    if use_tensorboard and tensorboard_writer is not None:
        for k, v in metrics.items():
            tensorboard_writer.add_scalar(f"{prefix}{k}", v, step)


def create_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Crear matriz de confusión.
    
    Args:
        predictions: Predicciones del modelo
        targets: Targets reales
        threshold: Umbral para binarización
        
    Returns:
        Matriz de confusión
    """
    # Aplicar sigmoid y binarizar
    if predictions.shape[1] == 1:
        probs = torch.sigmoid(predictions)
    else:
        probs = F.softmax(predictions, dim=1)
        probs = probs[:, 1:2]
    
    preds = (probs > threshold).float()
    
    # Convertir a numpy
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    
    # Calcular matriz de confusión
    tp = np.sum((preds_np == 1) & (targets_np == 1))
    tn = np.sum((preds_np == 0) & (targets_np == 0))
    fp = np.sum((preds_np == 1) & (targets_np == 0))
    fn = np.sum((preds_np == 0) & (targets_np == 1))
    
    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Graficar matriz de confusión.
    
    Args:
        confusion_matrix: Matriz de confusión
        save_path: Ruta para guardar gráfico
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Damage', 'Damage'],
        yticklabels=['No Damage', 'Damage']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matriz de confusión guardada en {save_path}")
    
    plt.show() 