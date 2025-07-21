import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import get_dataloaders_aug

def tensor_to_rgb(img_tensor):
    """
    Convierte un tensor (C,H,W) [0,1] o [0,255] a imagen RGB para mostrar.
    Usa las 3 primeras bandas, o repite la primera si hay menos.
    """
    img = img_tensor.detach().cpu().numpy()
    C, H, W = img.shape
    if C < 3:
        # Repite la primera banda hasta tener 3 canales
        img = np.tile(img, (3 // C + 1, 1, 1))[:3, :, :]
    elif C > 3:
        img = img[:3, :, :]
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualiza imágenes augmentadas de un batch del DataLoader")
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración YAML')
    parser.add_argument('--n', type=int, default=4, help='Número de imágenes a mostrar del batch')
    parser.add_argument('--repeat', type=int, default=3, help='Cuántas veces mostrar el mismo batch para ver augmentations')
    args = parser.parse_args()

    # Cargar config
    from config import Config
    config = Config.from_yaml(args.config)

    # Obtener DataLoader con augmentation, pero sin collation (para permitir diferentes shapes)
    from torch.utils.data import DataLoader
    train_loader, _, _ = get_dataloaders_aug(config)
    # Creamos un DataLoader especial para visualización
    vis_loader = DataLoader(
        train_loader.dataset,  # AugmentedDataset
        batch_size=args.n,
        shuffle=True,
        num_workers=0,  # Para debug, mejor 0
        collate_fn=lambda x: x  # Devuelve lista de samples
    )

    for rep in range(args.repeat):
        batch = next(iter(vis_loader))  # batch es una lista de dicts
        print(f"Batch {rep+1}:")
        plt.figure(figsize=(4*args.n, 4))
        for i, sample in enumerate(batch):
            img = tensor_to_rgb(sample['image'])
            label = int(sample['damaged'].item())
            fid = sample['fid']
            plt.subplot(1, args.n, i+1)
            plt.imshow(img)
            plt.title(f"FID: {fid}\nLabel: {label}")
            plt.axis('off')
        plt.suptitle(f"Augmented batch #{rep+1}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()