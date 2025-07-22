"""
Módulo de carga de datos para el proyecto AI for Earthquake Response.

Este módulo implementa la carga y preprocesamiento de imágenes satelitales
multitemporales (pre y post-terremoto) y sus máscaras de daño asociadas.
Adaptado para la nueva estructura de datos generada por preprocess.py.
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from rich.console import Console
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NUEVO: Importar sampler_aug para muestreo balanceado y aug adaptativo ---
from sampler_aug import make_balanced_sampler, build_transforms, AugmentedDataset

class EarthquakeTileDataset(Dataset):
    """
    Dataset para tiles preprocesados con estructura pre/post evento.
    Adaptado para la nueva estructura de datos generada por preprocess.py.
    """
    def __init__(self, data_root: str, split: str = 'full', mode: str = 'pre_post', 
                 filter_has_data: bool = True, max_channels: int = None):
        """
        Args:
            data_root: Directorio raíz de datos preprocesados
            split: 'full', 'test', o 'unknown'
            mode: 'pre_post' (ambos), 'pre_only', 'post_only'
            filter_has_data: Si True, solo incluye tiles con datos disponibles
            max_channels: Máximo número de canales a usar (None = todos)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.filter_has_data = filter_has_data
        self.max_channels = max_channels
        
        # Cargar metadatos
        meta_path = self.data_root / "meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"No se encontró {meta_path}")
        
        self.meta_df = pd.read_csv(meta_path)
        
        # Filtrar por split
        self.meta_df = self.meta_df[self.meta_df['split'] == split].copy()
        
        # Filtrar por disponibilidad de datos según modo
        if filter_has_data:
            if mode == 'pre_post':
                # Incluir tiles que tengan al menos pre o post
                self.meta_df = self.meta_df[
                    (self.meta_df['has_pre'] == 1) | (self.meta_df['has_post'] == 1)
                ]
            elif mode == 'pre_only':
                self.meta_df = self.meta_df[self.meta_df['has_pre'] == 1]
            elif mode == 'post_only':
                self.meta_df = self.meta_df[self.meta_df['has_post'] == 1]
        
        # Resetear índice
        self.meta_df = self.meta_df.reset_index(drop=True)
        
        logger.info(f"Dataset cargado: {len(self.meta_df)} tiles para split '{split}', modo '{mode}'")
        logger.info(f"Tiles con pre: {self.meta_df['has_pre'].sum()}")
        logger.info(f"Tiles con post: {self.meta_df['has_post'].sum()}")
        logger.info(f"Tiles dañados: {(self.meta_df['damaged'] == 1).sum()}")

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        
        # Cargar tile
        tile_path = Path(row['tile_path'])
        if not tile_path.exists():
            raise FileNotFoundError(f"No se encontró el tile: {tile_path}")
        
        tile_data = np.load(tile_path)
        # --- Chequeo robusto para npz/npy ---
        if isinstance(tile_data, np.lib.npyio.NpzFile):
            has_pre = 'pre' in tile_data
            has_post = 'post' in tile_data
        else:
            has_pre = has_post = False
        
        if has_pre:
            pre_data = tile_data['pre']  # (C, H, W)
        else:
            # Si no hay datos pre, crear array de ceros del tamaño correcto
            # Usar el shape de post si existe, o un tamaño por defecto
            if has_post:
                shape = tile_data['post'].shape
            else:
                shape = (3, 256, 256)  # tamaño por defecto
            pre_data = np.zeros(shape, dtype=np.float32)
            
        if has_post:
            post_data = tile_data['post']  # (C, H, W)
        else:
            # Si no hay datos post, usar el mismo shape que pre
            shape = pre_data.shape
            post_data = np.zeros(shape, dtype=np.float32)
        
        # Limitar canales si se especifica
        if self.max_channels is not None:
            pre_data = pre_data[:self.max_channels]
            post_data = post_data[:self.max_channels]
        # Si max_channels es None, usar todos los canales (no recortar)
        
        # Preparar imagen según modo
        if self.mode == 'pre_post':
            image = np.concatenate([pre_data, post_data], axis=0)
            # Crear máscara de presencia para pre_post
            mask = np.concatenate([
                np.ones(pre_data.shape[0]) if has_pre else np.zeros(pre_data.shape[0]),
                np.ones(post_data.shape[0]) if has_post else np.zeros(post_data.shape[0])
            ])
            # Expandir la máscara a (C_total, H, W)
            mask = mask[:, None, None].astype(np.float32)
            mask = np.broadcast_to(mask, image.shape).copy()
            assert mask.shape == image.shape, f"Presence mask shape {mask.shape} != image shape {image.shape}"
        elif self.mode == 'pre_only':
            image = pre_data
            mask = np.ones(pre_data.shape[0]) if has_pre else np.zeros(pre_data.shape[0])
        elif self.mode == 'post_only':
            image = post_data
            mask = np.ones(post_data.shape[0]) if has_post else np.zeros(post_data.shape[0])
        else:
            raise ValueError(f"Modo no soportado: {self.mode}")
        
        # --- Rellenar con ceros si el número de canales es menor al esperado ---
        if self.max_channels is not None:
            expected_channels = self.max_channels
            if self.mode == 'pre_post':
                expected_channels *= 2
        else:
            expected_channels = image.shape[0]
        
        if image.shape[0] < expected_channels:
            pad = np.zeros((expected_channels - image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
            image = np.concatenate([image, pad], axis=0)
            # Extender la máscara con ceros para los canales añadidos
            faltan = expected_channels - mask.shape[0]
            mask = np.concatenate([mask, np.zeros((faltan, image.shape[1], image.shape[2]), dtype=mask.dtype)], axis=0)
        elif image.shape[0] > expected_channels:
            image = image[:expected_channels]
            mask = mask[:expected_channels]
        
        # Normalización robusta según dtype
        if image.dtype == np.uint16:
            image = image / 65535.0
        elif image.dtype == np.uint8:
            image = image / 255.0
        # Si es float, asume que ya está normalizada
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        # --- Opcional: expandir presence_mask a (C, H, W) si se usa como canal ---
        # mask = mask[:, None, None].expand(-1, image.shape[1], image.shape[2])
        
        # Preparar label
        damaged = int(row['damaged'])
        if damaged == -1:  # NaN o sin etiqueta
            damaged = 0  # Asumir no dañado
        
        return {
            'image': image,
            'presence_mask': mask,  # Si quieres shape (C,H,W), descomenta la línea arriba
            'damaged': torch.tensor(damaged, dtype=torch.float32),
            'fid': row['fid'],
            'tile_path': str(tile_path),
            'event': row['event'],
            'has_pre': row['has_pre'],
            'has_post': row['has_post']
        }

class SubsetEarthquakeTileDataset(EarthquakeTileDataset):
    """Subset del dataset para train/val splits"""
    def __init__(self, data_root, split, mode, filter_has_data, max_channels, meta_subset):
        super().__init__(data_root, split, mode, filter_has_data, max_channels)
        self.meta_df = meta_subset

# --- NUEVO: Dataset wrapper para repetir datos (oversampling virtual) ---
class RepeatDataset(torch.utils.data.Dataset):
    """
    Dataset que repite el dataset base N veces (útil para oversampling/augmentation).
    """
    def __init__(self, base_dataset, repeat_factor=2):
        self.base_dataset = base_dataset
        self.repeat_factor = repeat_factor

    def __len__(self):
        return len(self.base_dataset) * self.repeat_factor

    def __getitem__(self, idx):
        return self.base_dataset[idx % len(self.base_dataset)]

# Ejemplo de uso en get_dataloaders_aug o get_dataloaders:
# repeat_factor = getattr(config.data, 'repeat_factor', 1)
# if repeat_factor > 1:
#     dataset_train_aug = RepeatDataset(dataset_train_aug, repeat_factor=repeat_factor)

def custom_collate(batch):
    """Función de collate personalizada para manejar tensores de diferentes tamaños"""
    batch_out = {}
    for key in batch[0].keys():
        if key in ['image']:
            # Apilar imágenes
            batch_out[key] = torch.stack([d[key] for d in batch])
        elif key in ['presence_mask']:
            # Apilar máscaras de presencia
            masks = [d[key] for d in batch]
            # Si alguna máscara es 3D (C, H, W), apila normalmente
            # Si alguna es 1D (C,), expande a (C, 1, 1) y luego apila
            if masks[0].ndim == 1:
                masks = [m[:, None, None] for m in masks]
            batch_out[key] = torch.stack(masks)
        elif key in ['damaged']:
            # Apilar labels
            batch_out[key] = torch.stack([d[key] for d in batch])
        else:
            # Mantener como lista para otros campos
            batch_out[key] = [d[key] for d in batch]
    return batch_out

def create_train_val_splits(dataset, split_ratio=0.8, seed=42):
    """Crear splits de train/val"""
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    n_train = int(len(indices) * split_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Crear metadatos para cada split
    meta_train = dataset.meta_df.iloc[train_indices].reset_index(drop=True)
    meta_val = dataset.meta_df.iloc[val_indices].reset_index(drop=True)
    
    return meta_train, meta_val

# --- NUEVA FUNCIÓN: get_dataloaders_aug ---
def get_dataloaders_aug(config):
    """
    Crea DataLoaders usando EarthquakeTileDataset y aplica:
    - Sampler balanceado y augmentations adaptativos SOLO en entrenamiento
    - Validación y test sin sampler ni augmentations
    Split de train/val a nivel de evento (no aleatorio por tile).
    Recibe un objeto config (como en train.py)
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    import warnings
    data_root = getattr(config.data, 'data_root', 'preprocessed')
    split = getattr(config.data, 'split', 'full')
    mode = getattr(config.data, 'mode', 'post_only')
    filter_has_data = getattr(config.data, 'filter_has_data', True)
    max_channels = getattr(config.data, 'max_channels', 3)
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    pin_memory = getattr(config.data, 'pin_memory', True)
    split_ratio = getattr(config.data, 'split_ratio', 0.8)
    seed = getattr(config, 'seed', 42)
    repeat_factor = getattr(config.data, 'repeat_factor', 1)

    # Cargar dataset completo
    dataset_full = EarthquakeTileDataset(
        data_root=data_root,
        split=split,
        mode=mode,
        filter_has_data=filter_has_data,
        max_channels=max_channels
    )
    meta_df = dataset_full.meta_df.copy()
    # --- Split estratificado por clase (no por evento) ---
    indices = np.arange(len(meta_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1-split_ratio,
        random_state=seed,
        stratify=meta_df['damaged']
    )
    meta_train = meta_df.iloc[train_idx].reset_index(drop=True)
    meta_val = meta_df.iloc[val_idx].reset_index(drop=True)
    # Crear datasets para train y val
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
    # --- Balanceo y augmentations SOLO para entrenamiento ---
    labels = dataset_train.meta_df["damaged"].astype(int).values
    n0, n1 = np.bincount(labels)
    minority = 0 if n0 < n1 else 1
    sampler = make_balanced_sampler(labels)
    tr_map = build_transforms(minority)
    dataset_train_aug = AugmentedDataset(dataset_train, tr_map)
    # --- Repetir el dataset si repeat_factor > 1 ---
    if repeat_factor > 1:
        dataset_train_aug = RepeatDataset(dataset_train_aug, repeat_factor=repeat_factor)
    train_loader = DataLoader(
        dataset_train_aug,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate
    )
    # --- Validación y test SIN sampler ni aug ---
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate
    )
    # Test loader si existe split test
    try:
        dataset_test = EarthquakeTileDataset(
            data_root=data_root,
            split='test',
            mode=mode,
            filter_has_data=filter_has_data,
            max_channels=max_channels
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate
        )
    except Exception:
        test_loader = None
    return train_loader, val_loader, test_loader

# --- NUEVA FUNCIÓN: get_dataloaders (sin augmentation, solo split por evento) ---
def get_dataloaders(config):
    """
    Crea DataLoaders usando EarthquakeTileDataset y split por evento,
    pero SIN sampler balanceado ni augmentations.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    data_root = getattr(config.data, 'data_root', 'preprocessed')
    split = getattr(config.data, 'split', 'full')
    mode = getattr(config.data, 'mode', 'post_only')
    filter_has_data = getattr(config.data, 'filter_has_data', True)
    max_channels = getattr(config.data, 'max_channels', 3)
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    pin_memory = getattr(config.data, 'pin_memory', True)
    split_ratio = getattr(config.data, 'split_ratio', 0.8)
    seed = getattr(config, 'seed', 42)

    dataset_full = EarthquakeTileDataset(
        data_root=data_root,
        split=split,
        mode=mode,
        filter_has_data=filter_has_data,
        max_channels=max_channels
    )
    meta_df = dataset_full.meta_df.copy()
    # --- Split estratificado por clase (no por evento) ---
    indices = np.arange(len(meta_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=1-split_ratio,
        random_state=seed,
        stratify=meta_df['damaged']
    )
    meta_train = meta_df.iloc[train_idx].reset_index(drop=True)
    meta_val = meta_df.iloc[val_idx].reset_index(drop=True)
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
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate
    )
    # Test loader si existe split test
    try:
        dataset_test = EarthquakeTileDataset(
            data_root=data_root,
            split='test',
            mode=mode,
            filter_has_data=filter_has_data,
            max_channels=max_channels
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate
        )
    except Exception:
        test_loader = None
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description="Prepara datasets y dataloaders para el proyecto de terremotos")
    parser.add_argument('--data_root', type=str, default='preprocessed', 
                       help='Directorio raíz de los datos preprocesados')
    parser.add_argument('--split', type=str, default='full', choices=['full', 'test', 'unknown'],
                       help='Split a usar')
    parser.add_argument('--mode', type=str, default='pre_only', 
                       choices=['pre_post', 'pre_only', 'post_only'],
                       help='Modo de datos: pre_post, pre_only, post_only')
    parser.add_argument('--filter_has_data', action='store_true', default=False,
                       help='Filtrar tiles sin datos disponibles')
    parser.add_argument('--max_channels', type=int, default=3,
                       help='Máximo número de canales a usar')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Tamaño de batch para el DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, 
                       help='Número de workers para el DataLoader')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Semilla para el split aleatorio')
    parser.add_argument('--split_ratio', type=float, default=0.8, 
                       help='Proporción de train (el resto es val)')
    parser.add_argument('--no_save', action='store_true',
                       help='No guardar datasets y dataloaders')
    parser.add_argument('--small_dataset', action='store_true',
                       help='Crear un dataset pequeño para pruebas rápidas')
    parser.add_argument('--small_percentage', type=float, default=0.2,
                       help='Porcentaje del dataset completo para crear dataset small (0.0-1.0)')
    parser.add_argument('--use_aug', action='store_true',default=True,
                        help='Usar muestreo balanceado y augmentations adaptativos (get_dataloaders_aug)')
    args = parser.parse_args()

    console.print(f"[bold cyan][INFO][/bold cyan] Configuración:")
    console.print(f"  Data root: [bold yellow]{args.data_root}[/bold yellow]")
    console.print(f"  Split: [bold yellow]{args.split}[/bold yellow]")
    console.print(f"  Mode: [bold yellow]{args.mode}[/bold yellow]")
    console.print(f"  Filter has data: [bold yellow]{args.filter_has_data}[/bold yellow]")
    console.print(f"  Max channels: [bold yellow]{args.max_channels}[/bold yellow]")
    if args.small_dataset:
        console.print(f"  Small dataset: [bold yellow]SÍ[/bold yellow] ({args.small_percentage*100:.1f}% del completo)")
    if args.use_aug:
        console.print(f"  [bold magenta]Usando get_dataloaders_aug (sampler balanceado + aug adaptativo)[/bold magenta]")

    if args.use_aug:
        # Crear un objeto de configuración mínimo para get_dataloaders_aug
        class DummyConfig:
            pass
        class DummyData:
            pass
        config = DummyConfig()
        config.data = DummyData()
        config.data.data_root = args.data_root
        config.data.split = args.split
        config.data.mode = args.mode
        config.data.filter_has_data = args.filter_has_data
        config.data.max_channels = args.max_channels
        config.data.batch_size = args.batch_size
        config.data.num_workers = args.num_workers
        config.data.pin_memory = True
        config.data.split_ratio = args.split_ratio
        config.seed = args.seed
        train_loader, val_loader, test_loader = get_dataloaders_aug(config)
        # Mensaje específico de número de tiles
        n_train = len(train_loader.dataset.ds) if hasattr(train_loader.dataset, 'ds') else len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_test = len(test_loader.dataset) if test_loader is not None else 0
        console.print(f"[bold green][INFO][/bold green] Tiles: Train: [bold]{n_train}[/bold] | Val: [bold]{n_val}[/bold] | Test: [bold]{n_test}[/bold]")
        console.print(f"[bold green][INFO][/bold green] DataLoaders (AUG) creados con batch_size=[bold]{args.batch_size}[/bold]")
        # Mostrar información de un batch de ejemplo
        try:
            sample_batch = next(iter(train_loader))
            console.print(f"[bold blue][INFO][/bold blue] Ejemplo de batch (AUG):")
            console.print(f"  Image shape: {sample_batch['image'].shape}")
            console.print(f"  Damaged labels: {sample_batch['damaged']}")
            console.print(f"  FIDs: {sample_batch['fid'][:3]}...")
            # --- PRINTS ADICIONALES PARA pre_post ---
            if args.mode == 'pre_post':
                pmask = sample_batch['presence_mask']
                console.print(f"  Presence mask shape: {pmask.shape}")
                console.print(f"  Presence mask sum por tile: {pmask.sum(dim=1)}")
                # Chequear si hay tiles con pre o post todo ceros
                nchan = pmask.shape[1] // 2
                pre_zeros = (sample_batch['image'][:, :nchan].abs().sum(dim=(1,2,3)) == 0)
                post_zeros = (sample_batch['image'][:, nchan:].abs().sum(dim=(1,2,3)) == 0)
                console.print(f"  Tiles con pre todo ceros: {pre_zeros.nonzero().flatten().tolist()}")
                console.print(f"  Tiles con post todo ceros: {post_zeros.nonzero().flatten().tolist()}")
        except Exception as e:
            console.print(f"[bold yellow][WARNING][/bold yellow] No se pudo mostrar ejemplo de batch: {e}")
        # Mostrar info de test si existe
        if test_loader is not None:
            try:
                test_batch = next(iter(test_loader))
                console.print(f"[bold blue][INFO][/bold blue] Ejemplo de batch de test:")
                console.print(f"  Image shape: {test_batch['image'].shape}")
                console.print(f"  FIDs: {test_batch['fid'][:3]}...")
                if args.mode == 'pre_post':
                    pmask = test_batch['presence_mask']
                    console.print(f"  Presence mask shape: {pmask.shape}")
                    console.print(f"  Presence mask sum por tile: {pmask.sum(dim=1)}")
                    nchan = pmask.shape[1] // 2
                    pre_zeros = (test_batch['image'][:, :nchan].abs().sum(dim=(1,2,3)) == 0)
                    post_zeros = (test_batch['image'][:, nchan:].abs().sum(dim=(1,2,3)) == 0)
                    console.print(f"  Tiles con pre todo ceros: {pre_zeros.nonzero().flatten().tolist()}")
                    console.print(f"  Tiles con post todo ceros: {post_zeros.nonzero().flatten().tolist()}")
            except Exception as e:
                console.print(f"[bold yellow][WARNING][/bold yellow] No se pudo mostrar ejemplo de batch de test: {e}")
        console.print(f"[bold green][SUCCESS][/bold green] Proceso finalizado correctamente (AUG)")
        return
    # --- flujo clásico (sin aug) ---

    # Cargar dataset base
    try:
        dataset_base = EarthquakeTileDataset(
            data_root=args.data_root,
            split=args.split,
            mode=args.mode,
            filter_has_data=args.filter_has_data,
            max_channels=args.max_channels
        )
        console.print(f"[bold green][INFO][/bold green] Dataset base cargado: [bold]{len(dataset_base)}[/bold] tiles")
    except Exception as e:
        console.print(f"[bold red][ERROR][/bold red] Error cargando dataset: {e}")
        return

    # Crear dataset small si se solicita
    if args.small_dataset:
        n_small = max(1, int(len(dataset_base) * args.small_percentage))
        np.random.seed(args.seed)
        small_indices = np.random.choice(len(dataset_base), n_small, replace=False)
        small_meta = dataset_base.meta_df.iloc[small_indices].reset_index(drop=True)
        
        # Crear dataset small
        dataset_base = SubsetEarthquakeTileDataset(
            data_root=args.data_root,
            split=args.split,
            mode=args.mode,
            filter_has_data=args.filter_has_data,
            max_channels=args.max_channels,
            meta_subset=small_meta
        )
        console.print(f"[bold green][INFO][/bold green] Dataset small creado: [bold]{len(dataset_base)}[/bold] tiles ({args.small_percentage*100:.1f}%)")

    # Crear splits train/val
    meta_train, meta_val = create_train_val_splits(dataset_base, args.split_ratio, args.seed)
    
    # Crear datasets para train y val
    dataset_train = SubsetEarthquakeTileDataset(
        data_root=args.data_root,
        split=args.split,
        mode=args.mode,
        filter_has_data=args.filter_has_data,
        max_channels=args.max_channels,
        meta_subset=meta_train
    )
    dataset_val = SubsetEarthquakeTileDataset(
        data_root=args.data_root,
        split=args.split,
        mode=args.mode,
        filter_has_data=args.filter_has_data,
        max_channels=args.max_channels,
        meta_subset=meta_val
    )
    
    console.print(f"[bold green][INFO][/bold green] Train: {len(dataset_train)} tiles | Val: {len(dataset_val)} tiles")
    n_test = 0
    try:
        test_dataset = EarthquakeTileDataset(
            data_root=args.data_root,
            split='test',
            mode=args.mode,
            filter_has_data=args.filter_has_data,
            max_channels=args.max_channels
        )
        n_test = len(test_dataset)
    except Exception:
        pass
    console.print(f"[bold green][INFO][/bold green] Tiles: Train: [bold]{len(dataset_train)}[/bold] | Val: [bold]{len(dataset_val)}[/bold] | Test: [bold]{n_test}[/bold]")

    # Crear dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    console.print(f"[bold green][INFO][/bold green] DataLoaders creados con batch_size=[bold]{args.batch_size}[/bold]")

    # Mostrar información de un batch de ejemplo
    try:
        sample_batch = next(iter(dataloader_train))
        console.print(f"[bold blue][INFO][/bold blue] Ejemplo de batch:")
        console.print(f"  Image shape: {sample_batch['image'].shape}")
        console.print(f"  Damaged labels: {sample_batch['damaged']}")
        console.print(f"  FIDs: {sample_batch['fid'][:3]}...")  # Primeros 3
    except Exception as e:
        console.print(f"[bold yellow][WARNING][/bold yellow] No se pudo mostrar ejemplo de batch: {e}")

    # Crear y mostrar dataloader de test si existe el split 'test'
    try:
        test_dataset = EarthquakeTileDataset(
            data_root=args.data_root,
            split='test',
            mode=args.mode,
            filter_has_data=args.filter_has_data,
            max_channels=args.max_channels
        )
        dataloader_test = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate
        )
        console.print(f"[bold green][INFO][/bold green] Test: {len(test_dataset)} tiles")
        try:
            test_batch = next(iter(dataloader_test))
            console.print(f"[bold blue][INFO][/bold blue] Ejemplo de batch de test:")
            console.print(f"  Image shape: {test_batch['image'].shape}")
            console.print(f"  FIDs: {test_batch['fid'][:3]}...")
        except Exception as e:
            console.print(f"[bold yellow][WARNING][/bold yellow] No se pudo mostrar ejemplo de batch de test: {e}")
    except Exception as e:
        console.print(f"[bold yellow][WARNING][/bold yellow] No se pudo crear dataloader de test: {e}")

    # Guardar datasets y dataloaders
    if not args.no_save:
        os.makedirs('outputs/datasets', exist_ok=True)
        os.makedirs('outputs/dataloaders', exist_ok=True)
        
        # Determinar sufijo para el nombre del archivo
        suffix = f"{args.split}_{args.mode}"
        if args.small_dataset:
            suffix += f"_small{args.small_percentage:.0%}"
        
        # Guardar datasets
        torch.save(dataset_train, f"outputs/datasets/dataset_train_{suffix}.pt")
        torch.save(dataset_val, f"outputs/datasets/dataset_val_{suffix}.pt")
        # Guardar solo los metadatos de los splits para reproducibilidad
        dataset_train.meta_df.to_csv(f"outputs/datasets/meta_train_{suffix}.csv", index=False)
        dataset_val.meta_df.to_csv(f"outputs/datasets/meta_val_{suffix}.csv", index=False)
        # Ya NO guardar DataLoader completos (no portable)
        # torch.save(dataloader_train, ...)
        # torch.save(dataloader_val, ...)
        # torch.save(dataloader_test, ...)
        console.print(f"[bold blue][INFO][/bold blue] Datasets y metadatos guardados en outputs/")
        console.print(f"[bold blue][INFO][/bold blue] Sufijo usado: {suffix}")
    
    console.print(f"[bold green][SUCCESS][/bold green] Proceso finalizado correctamente")

if __name__ == '__main__':
    main() 
    