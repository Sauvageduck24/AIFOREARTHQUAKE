"""
Módulo de carga de datos para el proyecto AI for Earthquake Response.

Este módulo implementa la carga y preprocesamiento de imágenes satelitales
multitemporales (pre y post-terremoto) y sus máscaras de daño asociadas.
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from rich.console import Console

console = Console()

class PreprocessedTileDatasetNPZ(Dataset):
    """
    Dataset para tiles preprocesados guardados como .npz y meta.json.
    Permite diferentes modos de entrenamiento según el parámetro 'mode'.
    """
    def __init__(self, data_root: str, split: str = 'full', with_mask: bool = True, mode: str = 'basic'):
        self.data_root = Path(data_root)
        self.split = split
        self.with_mask = with_mask
        self.mode = mode
        meta_path = self.data_root / split / 'meta.json'
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        self.images_dir = self.data_root / split / 'images'
        self.masks_dir = self.data_root / split / 'masks'

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        tile_id = item['id']
        img_path = self.images_dir / f"{tile_id}.npz"
        img_npz = np.load(img_path)
        image = img_npz[img_npz.files[0]]  # (C, H, W)
        if image.shape[0] > 6:
            image = image[:6]  # Solo los primeros 6 canales
        image = torch.from_numpy(image).float()
        # Modo básico: solo imagen y vector de labels (damaged)
        if self.mode == 'basic':
            # Target: vector de damaged para los fids presentes
            fids = item.get('fids', [])
            damaged_dict = item.get('damaged', {})
            # Ordenar los fids para que el vector de labels sea consistente
            fids_sorted = sorted(fids)
            labels = [damaged_dict.get(str(fid), None) for fid in fids_sorted]
            labels = torch.tensor(labels, dtype=torch.long)
            return {'image': image, 'labels': labels, 'fids': fids_sorted, 'tile_id': tile_id}
        # Modo intermedio: imagen, máscara de geometría y vector de labels
        elif self.mode == 'intermediate':
            # Máscara de geometría: 1 donde hay algún polígono, 0 fuera
            mask_path = self.masks_dir / f"{tile_id}.npz"
            mask_npz = np.load(mask_path)
            mask = mask_npz[mask_npz.files[0]]
            mask_geom = (mask > 0).astype(np.uint8)  # 1 donde hay edificio
            mask_geom = torch.from_numpy(mask_geom).long()
            fids = item.get('fids', [])
            damaged_dict = item.get('damaged', {})
            fids_sorted = sorted(fids)
            labels = [damaged_dict.get(str(fid), None) for fid in fids_sorted]
            labels = torch.tensor(labels, dtype=torch.long)
            return {'image': image, 'mask_geom': mask_geom, 'labels': labels, 'fids': fids_sorted, 'tile_id': tile_id}
        else:
            raise ValueError(f"Modo de dataset no soportado: {self.mode}")

# Subset serializable fuera de main
class SubsetPreprocessedTileDatasetNPZ(PreprocessedTileDatasetNPZ):
    def __init__(self, data_root, split, with_mask, mode, meta_subset):
        super().__init__(data_root, split, with_mask, mode)
        self.meta = meta_subset

def custom_collate(batch):
    # Devuelve listas de cada campo, sin apilar
    batch_out = {}
    for key in batch[0]:
        batch_out[key] = [d[key] for d in batch]
    return batch_out

def main():
    parser = argparse.ArgumentParser(description="Prepara y guarda datasets y dataloaders a partir de los tiles preprocesados.")
    parser.add_argument('--data_root', type=str, default='data/preprocessed', help='Directorio raíz de los datos preprocesados')
    parser.add_argument('--small', action='store_true', help='Si se usa, emplea el split small en vez de full',default=True)
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño de batch para el DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para el DataLoader')
    parser.add_argument('--mode', type=str, default='basic', choices=['basic', 'intermediate'], 
                        help='Modo de entrenamiento: basic (solo imagen), intermediate (imagen + máscara de geometría)')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para el split aleatorio')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Proporción de train (el resto es val)')
    args = parser.parse_args()

    split_base = 'small' if args.small else 'full'
    console.print(f"[bold cyan][INFO][/bold cyan] Usando split base: [bold yellow]{split_base}[/bold yellow]")
    console.print(f"[bold cyan][INFO][/bold cyan] Modo de entrenamiento: [bold yellow]{args.mode}[/bold yellow]")

    # Cargar dataset base
    dataset_base = PreprocessedTileDatasetNPZ(
        data_root=args.data_root,
        split=split_base,
        with_mask=True,
        mode=args.mode
    )
    console.print(f"[bold green][INFO][/bold green] Dataset base cargado: [bold]{len(dataset_base)}[/bold] tiles.")

    # Split aleatorio en train/val
    np.random.seed(args.seed)
    indices = np.arange(len(dataset_base))
    np.random.shuffle(indices)
    n_train = int(len(indices) * args.split_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Crear metadatas para cada split
    meta_train = [dataset_base.meta[i] for i in train_indices]
    meta_val = [dataset_base.meta[i] for i in val_indices]

    # Crear datasets para train y val
    dataset_train = SubsetPreprocessedTileDatasetNPZ(
        data_root=args.data_root,
        split=split_base,
        with_mask=True,
        mode=args.mode,
        meta_subset=meta_train
    )
    dataset_val = SubsetPreprocessedTileDatasetNPZ(
        data_root=args.data_root,
        split=split_base,
        with_mask=True,
        mode=args.mode,
        meta_subset=meta_val
    )
    console.print(f"[bold green][INFO][/bold green] Train: {len(dataset_train)} tiles | Val: {len(dataset_val)} tiles.")

    # Crear dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
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
    console.print(f"[bold green][INFO][/bold green] DataLoaders creados con batch_size=[bold]{args.batch_size}[/bold].")

    # Guardar datasets y dataloaders
    os.makedirs('data/datasets', exist_ok=True)
    os.makedirs('data/dataloaders', exist_ok=True)
    torch.save(dataset_train, f"data/datasets/dataset_train_{args.mode}.pt")
    torch.save(list(dataloader_train), f"data/dataloaders/dataloader_train_{args.mode}.pt")
    torch.save(dataset_val, f"data/datasets/dataset_val_{args.mode}.pt")
    torch.save(list(dataloader_val), f"data/dataloaders/dataloader_val_{args.mode}.pt")
    console.print(f"[bold blue][INFO][/bold blue] Datasets y dataloaders guardados en data/datasets y data/dataloaders.")
    console.print(f"[bold green][SUCCESS][/bold green] Proceso finalizado correctamente.")

if __name__ == '__main__':
    main() 
    