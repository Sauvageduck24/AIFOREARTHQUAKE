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
    """
    def __init__(self, data_root: str, split: str = 'full', with_mask: bool = True):
        self.data_root = Path(data_root)
        self.split = split
        self.with_mask = with_mask
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
        image = img_npz[img_npz.files[0]]  # 'arr_0'
        image = torch.from_numpy(image).float()
        if self.with_mask:
            mask_path = self.masks_dir / f"{tile_id}.npz"
            mask_npz = np.load(mask_path)
            mask = mask_npz[mask_npz.files[0]]
            mask = torch.from_numpy(mask).long()
        else:
            mask = None
        return {'image': image, 'mask': mask, 'tile_id': tile_id}


def main():
    parser = argparse.ArgumentParser(description="Prepara y guarda datasets y dataloaders a partir de los tiles preprocesados.")
    parser.add_argument('--data_root', type=str, default='data/preprocessed', help='Directorio raíz de los datos preprocesados')
    parser.add_argument('--small', action='store_true', help='Si se usa, emplea el split small en vez de full',default=True)
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño de batch para el DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para el DataLoader')
    args = parser.parse_args()

    split = 'small' if args.small else 'full'
    console.print(f"[bold cyan][INFO][/bold cyan] Usando split: [bold yellow]{split}[/bold yellow]")

    # Crear dataset
    dataset = PreprocessedTileDatasetNPZ(
        data_root=args.data_root,
        split=split,
        with_mask=True
    )
    console.print(f"[bold green][INFO][/bold green] Dataset cargado: [bold]{len(dataset)}[/bold] tiles.")

    # Crear dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    console.print(f"[bold green][INFO][/bold green] DataLoader creado con batch_size=[bold]{args.batch_size}[/bold].")

    # Guardar dataset y dataloader en disco
    os.makedirs('data/datasets', exist_ok=True)
    os.makedirs('data/dataloaders', exist_ok=True)
    dataset_path = f"data/datasets/dataset_{split}.pt"
    dataloader_path = f"data/dataloaders/dataloader_{split}.pt"
    torch.save(dataset, dataset_path)
    torch.save(list(dataloader), dataloader_path)
    console.print(f"[bold blue][INFO][/bold blue] Dataset guardado en: [bold]{dataset_path}[/bold]")
    console.print(f"[bold blue][INFO][/bold blue] Dataloader guardado en: [bold]{dataloader_path}[/bold]")
    console.print(f"[bold green][SUCCESS][/bold green] Proceso finalizado correctamente.")

if __name__ == '__main__':
    main() 
    