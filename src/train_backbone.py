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

# --- Augmentations optimizadas para SimCLR ---
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
    ], p=1.0)

# --- Augmentations más simples y rápidas ---
def get_fast_simclr_transform():
    return A.Compose([
        A.RandomResizedCrop((128, 128), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        ToTensorV2(),
    ], p=1.0)

# --- Dataset wrapper para SimCLR ---
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, transform, check_empty_images=False):
        self.ds = base_ds
        self.transform = transform
        self.check_empty_images = check_empty_images
        
        if check_empty_images:
            # Solo verificar si realmente es necesario (más eficiente)
            self.valid_indices = []
            print("[INFO] Verificando imágenes vacías/negras...")
            for i in tqdm(range(len(self.ds)), desc="Verificando imágenes"):
                try:
                    sample = self.ds[i]
                    img = sample['image']
                    if img is None:
                        continue
                    img_np = img.cpu().numpy() if torch.is_tensor(img) else img
                    # Verificación más rápida: solo verificar si la suma es muy baja
                    if img_np is not None and img_np.size > 0 and img_np.sum() > 1e-6:
                        self.valid_indices.append(i)
                except Exception as e:
                    print(f"Error procesando imagen {i}: {e}")
                    continue
            print(f"[INFO] Filtrando imágenes: {len(self.ds)} total, {len(self.valid_indices)} válidas")
        else:
            # Confiar en el filtro del dataset base
            self.valid_indices = list(range(len(self.ds)))
            print(f"[INFO] Usando todas las {len(self.ds)} imágenes del dataset base")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        sample = self.ds[original_idx]
        img = sample['image']
        
        # Optimización: evitar conversiones innecesarias si ya es tensor
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        
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
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (aumentado por defecto)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dim', type=int, default=128, help='Dimensión del proyector SimCLR')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='outputs/models/cnnbackbone_simclr.pth')
    parser.add_argument('--num_workers', type=int, default=min(os.cpu_count(), 8), help='Número de workers para DataLoader')
    parser.add_argument('--check_empty_images', action='store_true', help='Verificar imágenes vacías/negras (lento)')
    parser.add_argument('--max_samples', type=int, default=None, help='Máximo número de muestras para entrenamiento (para pruebas)')
    parser.add_argument('--splits', nargs='+', default=['full'], help='Splits a cargar')
    parser.add_argument('--fast_aug', action='store_true',default=True, help='Usar augmentaciones más rápidas')
    args = parser.parse_args()

    from torch.utils.data import ConcatDataset
    splits_to_load = args.splits
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
    
    # Limitar muestras si se especifica (útil para pruebas)
    if args.max_samples and len(base_ds) > args.max_samples:
        from torch.utils.data import Subset
        import random
        random.seed(42)
        indices = random.sample(range(len(base_ds)), args.max_samples)
        base_ds = Subset(base_ds, indices)
        print(f"[INFO] Limitando a {args.max_samples} muestras para entrenamiento")
    
    print(f"[INFO] Total de tiles para pre-entrenamiento: {len(base_ds)}")
    
    # Seleccionar transformaciones
    if args.fast_aug:
        transform = get_fast_simclr_transform()
        print("[INFO] Usando augmentaciones rápidas")
    else:
        transform = get_simclr_transform()
        print("[INFO] Usando augmentaciones completas")
    simclr_ds = SimCLRDataset(base_ds, transform, check_empty_images=args.check_empty_images)
    
    # Optimizar DataLoader
    loader = DataLoader(
        simclr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False,
        generator=torch.Generator(device='cpu')  # Para reproducibilidad
    )

    # Modelo
    backbone = CNNBackbone(in_channels=args.max_channels, dropout=0.1, base=64).to(args.device)
    projector = nn.Sequential(
        nn.Linear(backbone.out_dim, args.out_dim),
        nn.ReLU(),
        nn.Linear(args.out_dim, args.out_dim)
    ).to(args.device)
    
    # Optimizar optimizer
    optimizer = optim.AdamW(list(backbone.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    loss_fn = NTXentLoss(temperature=0.5)
    scaler = torch.cuda.amp.GradScaler()

    # Entrenamiento optimizado
    backbone.train()
    projector.train()
    all_epoch_losses = []
    
    print(f"[INFO] Iniciando entrenamiento con {len(loader)} batches por epoch")
    print(f"[INFO] Batch size: {args.batch_size}, Workers: {args.num_workers}")
    
    for epoch in range(args.epochs):
        total_loss = 0
        batch_losses = []
        running_loss = 0  # Para pérdida móvil
        
        # Optimizar barra de progreso con mejor formato
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                   leave=True, ncols=140, 
                   position=0, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (aug1, aug2) in enumerate(pbar):
            # Mover datos a GPU de forma más eficiente
            aug1 = aug1.to(args.device, non_blocking=True, dtype=torch.float16)
            aug2 = aug2.to(args.device, non_blocking=True, dtype=torch.float16)

            optimizer.zero_grad(set_to_none=True)  # Más eficiente
            
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
            
            # Calcular pérdida móvil (últimos 10 batches)
            running_loss = np.mean(batch_losses[-10:]) if len(batch_losses) >= 10 else np.mean(batch_losses)
            
            # Actualizar barra de progreso más frecuentemente con más información
            if batch_idx % 5 == 0 or batch_idx == len(loader) - 1:
                # Calcular tendencia de la pérdida
                if len(batch_losses) >= 20:
                    recent_losses = batch_losses[-20:]
                    trend = "↗️" if recent_losses[-1] > recent_losses[0] else "↘️" if recent_losses[-1] < recent_losses[0] else "➡️"
                else:
                    trend = "➡️"
                
                pbar.set_postfix({
                    'Loss': f"{loss_value:.4f}",
                    'Avg10': f"{running_loss:.4f}",
                    'EpochAvg': f"{np.mean(batch_losses):.4f}",
                    'Trend': trend,
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}",
                    'Min': f"{min(batch_losses):.4f}",
                    'Max': f"{max(batch_losses):.4f}"
                })
        
        scheduler.step()
        epoch_avg_loss = total_loss / len(loader)
        all_epoch_losses.append(epoch_avg_loss)
        
        # Mostrar resumen del epoch con más detalles
        print(f"\n[Epoch {epoch+1}] Resumen:")
        print(f"  • Pérdida promedio: {epoch_avg_loss:.4f}")
        print(f"  • Pérdida mínima: {min(batch_losses):.4f}")
        print(f"  • Pérdida máxima: {max(batch_losses):.4f}")
        print(f"  • Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        if len(all_epoch_losses) > 1:
            improvement = all_epoch_losses[-2] - epoch_avg_loss
            print(f"  • Mejora vs epoch anterior: {improvement:+.4f}")
        print("-" * 50)
    
    # Guardar backbone preentrenado
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(backbone.state_dict(), args.save_path)
    print(f"Backbone guardado en {args.save_path}")
    
    # Mostrar evolución completa de pérdidas
    print(f"\n📊 Evolución de pérdidas por epoch:")
    for i, loss in enumerate(all_epoch_losses):
        if i > 0:
            change = all_epoch_losses[i-1] - loss
            arrow = "↘️" if change > 0 else "↗️" if change < 0 else "➡️"
            print(f"  Epoch {i+1}: {loss:.4f} {arrow} ({change:+.4f})")
        else:
            print(f"  Epoch {i+1}: {loss:.4f}")

if __name__ == "__main__":
    main() 