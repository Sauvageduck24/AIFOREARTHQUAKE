import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import EarthquakeTileDataset, SubsetEarthquakeTileDataset, custom_collate
from model import CNNBackbone
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# & C:/Users/esteb/.conda/envs/earthquake/python.exe src/train_backbone.py --mode pre_post --max_channels 3 --epochs 10 --batch_size 32

# --- Augmentations fuertes estilo SimCLR ---
def get_simclr_transform():
    return A.Compose([
        A.RandomResizedCrop((128, 128), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussNoise(variance_limit=(10.0, 80.0), p=0.5),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
        ToTensorV2(),
    ])

# --- Dataset wrapper para SimCLR ---
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, transform):
        self.ds = base_ds
        self.transform = transform
        # Filtrar imágenes vacías, nulas o completamente negras
        self.valid_indices = []
        for i in range(len(self.ds)):
            try:
                sample = self.ds[i]
                img = sample['image']
                if img is None:
                    continue
                img_np = img.cpu().numpy() if torch.is_tensor(img) else img
                # Verificar si la imagen no está vacía y no es completamente negra
                if img_np is not None and img_np.size > 0 and not np.allclose(img_np, 0):
                    self.valid_indices.append(i)
            except Exception as e:
                print(f"Error procesando imagen {i}: {e}")
                continue
        print(f"[INFO] Filtrando imágenes: {len(self.ds)} total, {len(self.valid_indices)} válidas")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        sample = self.ds[original_idx]
        img = sample['image'].cpu().numpy()  # (C,H,W)
        img = np.transpose(img, (1,2,0))
        img = (img * 255).astype(np.uint8)
        # Dos vistas augmentadas
        aug1 = self.transform(image=img)['image'].float() / 255.
        aug2 = self.transform(image=img)['image'].float() / 255.
        return aug1, aug2

# --- NT-Xent Loss (SimCLR) ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N,2N)
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float('-inf'))
        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
        negatives = sim[~mask].view(2*N, -1)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2*N, dtype=torch.long, device=z.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

def main():
    parser = argparse.ArgumentParser(description='Preentrenamiento no supervisado del backbone CNN (SimCLR)')
    parser.add_argument('--data_root', type=str, default='preprocessed', help='Directorio de datos preprocesados')
    parser.add_argument('--mode', type=str, default='pre_post', choices=['pre_only','post_only','pre_post'], help='Modo de datos')
    parser.add_argument('--max_channels', type=int, default=3, help='Nº de canales a usar')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dim', type=int, default=128, help='Dimensión del proyector SimCLR')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='outputs/models/cnnbackbone_simclr.pth')
    parser.add_argument('--num_workers', type=int, default=min(os.cpu_count(), 12), help='Número de workers para DataLoader')
    args = parser.parse_args()

    from torch.utils.data import ConcatDataset
    splits_to_load = ['full', 'test']
    print(f"[INFO] Cargando datos de los splits: {splits_to_load}")

    # Dataset
    if args.mode == 'pre_post':
        all_datasets = []
        for split in splits_to_load:
            all_datasets.append(EarthquakeTileDataset(data_root=args.data_root, split=split, mode='pre_only', filter_has_data=True, max_channels=args.max_channels))
            all_datasets.append(EarthquakeTileDataset(data_root=args.data_root, split=split, mode='post_only', filter_has_data=True, max_channels=args.max_channels))
        base_ds = ConcatDataset(all_datasets)
    else:
        all_datasets = []
        for split in splits_to_load:
            all_datasets.append(EarthquakeTileDataset(data_root=args.data_root, split=split, mode=args.mode, filter_has_data=True, max_channels=args.max_channels))
        base_ds = ConcatDataset(all_datasets)
    
    print(f"[INFO] Total de tiles para pre-entrenamiento: {len(base_ds)}")
    
    transform = get_simclr_transform()
    simclr_ds = SimCLRDataset(base_ds, transform)
    loader = DataLoader(
        simclr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2
    )

    # Modelo
    backbone = CNNBackbone(in_channels=args.max_channels, dropout=0.1, base=64).to(args.device)
    projector = nn.Sequential(
        nn.Linear(backbone.out_dim, args.out_dim),
        nn.ReLU(),
        nn.Linear(args.out_dim, args.out_dim)
    ).to(args.device)
    optimizer = optim.Adam(list(backbone.parameters()) + list(projector.parameters()), lr=args.lr)
    loss_fn = NTXentLoss(temperature=0.5)
    scaler = torch.cuda.amp.GradScaler()

    # Entrenamiento
    backbone.train()
    projector.train()
    all_epoch_losses = []  # Guardar la pérdida media de cada epoch
    for epoch in range(args.epochs):
        total_loss = 0
        batch_losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True, ncols=100)
        for batch_idx, (aug1, aug2) in enumerate(pbar):
            aug1 = aug1.to(args.device, non_blocking=True)
            aug2 = aug2.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z1 = projector(backbone(aug1))
                z2 = projector(backbone(aug2))
                loss = loss_fn(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.item()
            total_loss += loss_value
            batch_losses.append(loss_value)
            # Actualizar barra de progreso con la pérdida media de la epoch y la del batch
            pbar.set_postfix({
                'BatchLoss': f"{loss_value:.4f}",
                'AvgLoss': f"{np.mean(batch_losses):.4f}"
            })
        epoch_avg_loss = total_loss / len(loader)
        all_epoch_losses.append(epoch_avg_loss)
        print(f"[Epoch {epoch+1}] Avg Loss: {epoch_avg_loss:.4f}")
    # Guardar backbone preentrenado solo al final
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(backbone.state_dict(), args.save_path)
    print(f"Backbone guardado en {args.save_path}")

if __name__ == "__main__":
    main() 