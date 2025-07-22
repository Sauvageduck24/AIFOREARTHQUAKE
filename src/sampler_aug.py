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
def build_transforms(minority_class=None):
    """Crea un único pipeline Albumentations para todas las clases."""
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15,
                           scale_limit=0.2,
                           rotate_limit=60, p=0.7),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.1),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        ToTensorV2(),
    ])
    return {0: aug, 1: aug}

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