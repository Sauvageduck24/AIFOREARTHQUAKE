"""
Módulo de carga de datos para el proyecto AI for Earthquake Response.

Este módulo implementa la carga y preprocesamiento de imágenes satelitales
multitemporales (pre y post-terremoto) y sus máscaras de daño asociadas.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import rasterio
from rasterio.windows import Window
import cv2
from loguru import logger
import json

from .config import Config
from .transforms import get_transforms

class EarthquakeDataset(Dataset):
    """
    Dataset personalizado para imágenes satelitales de terremotos.
    
    Este dataset maneja pares de imágenes pre/post-terremoto y sus
    máscaras de daño correspondientes.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
        image_size: int = 512,
        channels: int = 3,
        use_augmentation: bool = True
    ):
        """
        Inicializar el dataset.
        
        Args:
            data_root: Ruta raíz de los datos
            split: Partición del dataset ('train', 'val', 'test')
            transform: Transformaciones a aplicar
            image_size: Tamaño de las imágenes de salida
            channels: Número de canales de las imágenes
            use_augmentation: Si usar aumentación de datos
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.channels = channels
        self.use_augmentation = use_augmentation
        
        # Encontrar todas las escenas disponibles
        self.scenes = self._find_scenes()
        
        if len(self.scenes) == 0:
            raise ValueError(f"No se encontraron escenas en {data_root}")
        
        logger.info(f"Dataset {split}: {len(self.scenes)} escenas encontradas")
    
    def _find_scenes(self) -> List[Dict[str, str]]:
        """Encontrar todas las escenas disponibles en el directorio de datos."""
        scenes = []
        
        # Buscar directorios de escenas
        scene_dirs = glob.glob(str(self.data_root / "scene_*"))
        
        for scene_dir in scene_dirs:
            scene_path = Path(scene_dir)
            scene_name = scene_path.name
            
            # Verificar estructura de directorios
            pre_event_dir = scene_path / "pre_event"
            post_event_dir = scene_path / "post_event"
            annotations_dir = scene_path / "annotations"
            
            if not all([pre_event_dir.exists(), post_event_dir.exists(), annotations_dir.exists()]):
                logger.warning(f"Estructura incompleta en {scene_name}, saltando...")
                continue
            
            # Encontrar archivos de imágenes
            pre_images = self._find_image_files(pre_event_dir)
            post_images = self._find_image_files(post_event_dir)
            annotation_files = self._find_annotation_files(annotations_dir)
            
            if not pre_images or not post_images or not annotation_files:
                logger.warning(f"Archivos faltantes en {scene_name}, saltando...")
                continue
            
            # Crear entrada para cada combinación válida
            for pre_img, post_img, ann_file in zip(pre_images, post_images, annotation_files):
                scenes.append({
                    'scene_name': scene_name,
                    'pre_event_path': str(pre_img),
                    'post_event_path': str(post_img),
                    'annotation_path': str(ann_file)
                })
        
        return scenes
    
    def _find_image_files(self, directory: Path) -> List[Path]:
        """Encontrar archivos de imagen en un directorio."""
        extensions = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png']
        files = []
        
        for ext in extensions:
            files.extend(directory.glob(ext))
        
        return sorted(files)
    
    def _find_annotation_files(self, directory: Path) -> List[Path]:
        """Encontrar archivos de anotación en un directorio."""
        extensions = ['*.tif', '*.tiff', '*.png', '*.json']
        files = []
        
        for ext in extensions:
            files.extend(directory.glob(ext))
        
        return sorted(files)
    
    def __len__(self) -> int:
        """Retornar el número de muestras en el dataset."""
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Obtener una muestra del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Diccionario con imágenes y máscara
        """
        scene_info = self.scenes[idx]
        
        # Cargar imágenes pre y post evento
        pre_image = self._load_image(scene_info['pre_event_path'])
        post_image = self._load_image(scene_info['post_event_path'])
        
        # Cargar máscara de anotación
        mask = self._load_mask(scene_info['annotation_path'])
        
        # Concatenar imágenes pre y post
        combined_image = np.concatenate([pre_image, post_image], axis=2)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=combined_image, mask=mask)
            combined_image = transformed['image']
            mask = transformed['mask']
        
        # Convertir a tensores
        image_tensor = torch.from_numpy(combined_image).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # Añadir dimensión de canal
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'scene_name': scene_info['scene_name'],
            'pre_path': scene_info['pre_event_path'],
            'post_path': scene_info['post_event_path']
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Cargar imagen desde archivo."""
        try:
            # Intentar cargar con rasterio primero (para archivos GeoTIFF)
            with rasterio.open(image_path) as src:
                image = src.read()
                # Convertir de (C, H, W) a (H, W, C)
                image = np.transpose(image, (1, 2, 0))
                
                # Normalizar a [0, 1]
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) / 65535.0
                elif image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                
                # Asegurar que tenemos el número correcto de canales
                if image.shape[2] > self.channels:
                    image = image[:, :, :self.channels]
                elif image.shape[2] < self.channels:
                    # Repetir el último canal si es necesario
                    last_channel = image[:, :, -1:]
                    while image.shape[2] < self.channels:
                        image = np.concatenate([image, last_channel], axis=2)
                
                return image
                
        except Exception as e:
            # Fallback a PIL/OpenCV
            logger.warning(f"No se pudo cargar {image_path} con rasterio: {e}")
            
            if image_path.lower().endswith(('.tif', '.tiff')):
                # Para archivos TIFF que no son GeoTIFF
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"No se pudo cargar la imagen: {image_path}")
                
                # Convertir BGR a RGB si es necesario
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Usar PIL para otros formatos
                image = np.array(Image.open(image_path))
            
            # Normalizar
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Asegurar formato correcto
            if len(image.shape) == 2:
                image = np.stack([image] * self.channels, axis=2)
            elif image.shape[2] > self.channels:
                image = image[:, :, :self.channels]
            
            return image
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Cargar máscara de anotación."""
        try:
            # Intentar cargar con rasterio primero
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Leer solo el primer canal
                
                # Normalizar a [0, 1]
                if mask.dtype == np.uint16:
                    mask = mask.astype(np.float32) / 65535.0
                elif mask.dtype == np.uint8:
                    mask = mask.astype(np.float32) / 255.0
                
                return mask
                
        except Exception as e:
            # Fallback a PIL/OpenCV
            logger.warning(f"No se pudo cargar {mask_path} con rasterio: {e}")
            
            if mask_path.lower().endswith(('.tif', '.tiff')):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.array(Image.open(mask_path).convert('L'))
            
            if mask is None:
                raise ValueError(f"No se pudo cargar la máscara: {mask_path}")
            
            # Normalizar a [0, 1]
            if mask.dtype == np.uint16:
                mask = mask.astype(np.float32) / 65535.0
            elif mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0
            
            return mask


class PreprocessedTileDataset(Dataset):
    """
    Dataset para datos preprocesados por preprocess.py (tiles en .pt y meta.json).
    Permite filtrar por train/test usando el campo label_path.
    """
    def __init__(self, data_root: str, split: str = 'full', mode: str = 'all', transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.mode = mode
        meta_path = self.data_root / split / 'meta.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # Filtrar según el modo
        if mode == 'train':
            self.meta = [item for item in meta if 'training' in item.get('label_path', '').lower()]
        elif mode == 'test':
            self.meta = [item for item in meta if 'testing' in item.get('label_path', '').lower()]
        else:
            self.meta = meta
        self.images_dir = self.data_root / split / 'images'
        self.masks_dir = self.data_root / split / 'masks'

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        tile_id = item['id']
        img_path = self.images_dir / f"{tile_id}.pt"
        mask_path = self.masks_dir / f"{tile_id}.pt"
        image = torch.load(img_path, map_location='cpu')
        mask = torch.load(mask_path, map_location='cpu')
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out['image']
            mask = out['mask']
        return {
            'image': image,
            'mask': mask,
            'tile_id': tile_id,
            'city': item.get('city', ''),
            'scene_id': item.get('scene_id', ''),
            'bbox': item.get('bbox', None)
        }


def get_data_loaders(
    config: Config,
    data_root: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crear DataLoaders para train, validation y test.
    
    Args:
        config: Configuración del proyecto
        data_root: Ruta raíz de los datos (opcional)
        
    Returns:
        Tupla con DataLoaders para train, val y test
    """
    if data_root is None:
        data_root = config.data.raw_data_path
    
    # Obtener transformaciones
    train_transforms = get_transforms(
        split="train",
        image_size=config.data.image_size,
        use_augmentation=config.data.use_augmentation
    )
    
    val_transforms = get_transforms(
        split="val",
        image_size=config.data.image_size,
        use_augmentation=False
    )
    
    test_transforms = get_transforms(
        split="test",
        image_size=config.data.image_size,
        use_augmentation=False
    )
    
    # Crear dataset completo
    full_dataset = EarthquakeDataset(
        data_root=data_root,
        split="full",
        transform=train_transforms,
        image_size=config.data.image_size,
        channels=config.data.channels,
        use_augmentation=config.data.use_augmentation
    )
    
    # Calcular tamaños de splits
    total_size = len(full_dataset)
    train_size = int(config.data.train_split * total_size)
    val_size = int(config.data.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Crear splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Aplicar transformaciones específicas
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = test_transforms
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False
    )
    
    logger.info(f"DataLoaders creados:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(data_root: str) -> Dict[str, any]:
    """
    Obtener información sobre el dataset.
    
    Args:
        data_root: Ruta raíz de los datos
        
    Returns:
        Diccionario con información del dataset
    """
    dataset = EarthquakeDataset(data_root, split="full")
    
    info = {
        'total_scenes': len(dataset.scenes),
        'image_size': dataset.image_size,
        'channels': dataset.channels,
        'scenes': [scene['scene_name'] for scene in dataset.scenes]
    }
    
    return info 

if __name__ == '__main__':
    import time
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description="Test DataLoader standalone.")
    parser.add_argument('--data_root', type=str, default='data/preprocessed', help='Directorio raíz de los datos preprocesados')
    parser.add_argument('--split', type=str, default='full', help='Split a usar (full/small/val)')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño de batch para el DataLoader')
    parser.add_argument('--num_workers', type=int, default=2, help='Número de workers para el DataLoader')
    args = parser.parse_args()

    start = time.time()
    # Dataset y DataLoader para datos preprocesados (ALL)
    dataset = PreprocessedTileDataset(
        data_root=args.data_root,
        split=args.split,
        mode='all',
        transform=None
    )
    print(f"[INFO] PreprocessedTileDataset (ALL) cargado: {len(dataset)} tiles.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"[INFO] DataLoader (ALL) creado con batch_size={args.batch_size}.")

    # Guardar en memoria optimizada (solo los tensores)
    all_images = []
    all_masks = []
    for batch in dataloader:
        all_images.append(batch['image'])
        all_masks.append(batch['mask'])
    images_tensor = torch.cat(all_images, dim=0)
    masks_tensor = torch.cat(all_masks, dim=0)
    print(f"[INFO] Tensores concatenados (ALL): imágenes {images_tensor.shape}, máscaras {masks_tensor.shape}")

    # Guardar en archivos temporales
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f_img, tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f_mask:
        torch.save(images_tensor, f_img.name)
        torch.save(masks_tensor, f_mask.name)
        print(f"[INFO] Imágenes (ALL) guardadas en: {f_img.name}")
        print(f"[INFO] Máscaras (ALL) guardadas en: {f_mask.name}")

    # Dataset y DataLoader para TRAIN
    train_dataset = PreprocessedTileDataset(
        data_root=args.data_root,
        split=args.split,
        mode='train',
        transform=None
    )
    print(f"[INFO] PreprocessedTileDataset (TRAIN) cargado: {len(train_dataset)} tiles.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    all_train_images = []
    all_train_masks = []
    for batch in train_loader:
        all_train_images.append(batch['image'])
        all_train_masks.append(batch['mask'])
    if all_train_images:
        train_images_tensor = torch.cat(all_train_images, dim=0)
        train_masks_tensor = torch.cat(all_train_masks, dim=0)
        print(f"[INFO] Tensores concatenados (TRAIN): imágenes {train_images_tensor.shape}, máscaras {train_masks_tensor.shape}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f_img, tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f_mask:
            torch.save(train_images_tensor, f_img.name)
            torch.save(train_masks_tensor, f_mask.name)
            print(f"[INFO] Imágenes (TRAIN) guardadas en: {f_img.name}")
            print(f"[INFO] Máscaras (TRAIN) guardadas en: {f_mask.name}")
    else:
        print("[INFO] No hay datos de TRAIN para guardar.")

    # Dataset y DataLoader para TEST
    test_dataset = PreprocessedTileDataset(
        data_root=args.data_root,
        split=args.split,
        mode='test',
        transform=None
    )
    print(f"[INFO] PreprocessedTileDataset (TEST) cargado: {len(test_dataset)} tiles.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    all_test_images = []
    all_test_masks = []
    for batch in test_loader:
        all_test_images.append(batch['image'])
        all_test_masks.append(batch['mask'])
    if all_test_images:
        test_images_tensor = torch.cat(all_test_images, dim=0)
        test_masks_tensor = torch.cat(all_test_masks, dim=0)
        print(f"[INFO] Tensores concatenados (TEST): imágenes {test_images_tensor.shape}, máscaras {test_masks_tensor.shape}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f_img, tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f_mask:
            torch.save(test_images_tensor, f_img.name)
            torch.save(test_masks_tensor, f_mask.name)
            print(f"[INFO] Imágenes (TEST) guardadas en: {f_img.name}")
            print(f"[INFO] Máscaras (TEST) guardadas en: {f_mask.name}")
    else:
        print("[INFO] No hay datos de TEST para guardar.")

    elapsed = time.time() - start
    print(f"[INFO] Tiempo total: {elapsed:.2f} segundos.") 
    