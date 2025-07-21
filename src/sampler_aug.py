import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------- 1. Construir sampler balanceado ----------
def make_balanced_sampler(labels):
    """Devuelve un WeightedRandomSampler con pesos inversamente proporcionales
    al nº de ejemplos de cada clase."""
    class_sample_count = np.bincount(labels)        # [n_0, n_1]
    weights = 1. / class_sample_count               # peso ∝ 1/frecuencia
    sample_weights = torch.as_tensor(weights[labels], dtype=torch.double)  # FIX: ensure tensor
    return WeightedRandomSampler(sample_weights,
                                 num_samples=len(labels),
                                 replacement=True)

# ---------- 2. Data augmentation específico por clase ----------
def build_transforms(minority_class):
    """Crea dos pipelines Albumentations:
       - strong_aug   → se aplica a la clase minoritaria
       - weak_aug     → se aplica a la clase mayoritaria
    """
    strong_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.15,
                           rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GridDistortion(p=0.2),
        ToTensorV2(),
    ])
    weak_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        ToTensorV2(),
    ])
    # Mapear índice de clase → transform correspondiente
    tr_map = {minority_class: strong_aug,
              1 - minority_class: weak_aug}
    return tr_map

# ---------- 3. Dataset wrapper que inyecta la aug ----------
class AugmentedDataset(torch.utils.data.Dataset):
    """
    Envuelve tu EarthquakeTileDataset y aplica transformaciones
    distintas según la clase.
    """
    def __init__(self, base_ds, tr_map):
        self.ds = base_ds
        self.tr_map = tr_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]            # dict de data_loader.py
        img = sample["image"].cpu().contiguous().numpy()    # (C,H,W) float32 0-1, FIX: ensure CPU/contiguous
        lbl = int(sample["damaged"].item())

        # Albumentations espera (H,W,C)
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)

        # Chequeo defensivo: si lbl no está en tr_map, usar el menor key
        if lbl not in self.tr_map:
            lbl = min(self.tr_map.keys())
        img = self.tr_map[lbl](image=img)["image"]   # Tensor CxHxW
        # --- Asegura float32 [0,1] para el modelo ---
        img = img.float()
        if img.max() > 1:
            img = img / 255.
        sample["image"] = img
        return sample 