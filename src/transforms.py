"""
Módulo de transformaciones y aumentación de datos para el proyecto AI for Earthquake Response.

Este módulo implementa transformaciones específicas para imágenes satelitales
multitemporales, incluyendo aumentación de datos con albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional
import numpy as np
from loguru import logger


def get_transforms(
    split: str = "train",
    image_size: int = 512,
    use_augmentation: bool = True,
    augmentation_prob: float = 0.5
) -> A.Compose:
    """
    Obtener transformaciones para un split específico.
    
    Args:
        split: Split del dataset ('train', 'val', 'test')
        image_size: Tamaño de las imágenes de salida
        use_augmentation: Si usar aumentación de datos
        augmentation_prob: Probabilidad de aplicar aumentación
        
    Returns:
        Pipeline de transformaciones de albumentations
    """
    
    if split == "train" and use_augmentation:
        return get_train_transforms(image_size, augmentation_prob)
    else:
        return get_val_test_transforms(image_size)


def get_train_transforms(image_size: int, prob: float = 0.5) -> A.Compose:
    """
    Obtener transformaciones para entrenamiento con aumentación.
    
    Args:
        image_size: Tamaño de las imágenes de salida
        prob: Probabilidad de aplicar aumentación
        
    Returns:
        Pipeline de transformaciones para entrenamiento
    """
    
    return A.Compose([
        # Transformaciones geométricas
        A.RandomRotate90(p=prob),
        A.Flip(p=prob),
        A.Transpose(p=prob),
        
        # Transformaciones de escala y rotación
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=prob
        ),
        
        # Transformaciones de perspectiva
        A.Perspective(
            scale=(0.05, 0.1),
            p=prob * 0.5
        ),
        
        # Transformaciones de brillo y contraste
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1.0
            ),
        ], p=prob),
        
        # Transformaciones de color
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0
            ),
        ], p=prob * 0.5),
        
        # Transformaciones de ruido
        A.OneOf([
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=1.0
            ),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=1.0
            ),
        ], p=prob * 0.3),
        
        # Transformaciones de blur
        A.OneOf([
            A.MotionBlur(
                blur_limit=3,
                p=1.0
            ),
            A.MedianBlur(
                blur_limit=3,
                p=1.0
            ),
            A.Blur(
                blur_limit=3,
                p=1.0
            ),
        ], p=prob * 0.3),
        
        # Transformaciones específicas para imágenes satelitales
        A.OneOf([
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=1.0
            ),
        ], p=prob * 0.2),
        
        # Redimensionar
        A.Resize(
            height=image_size,
            width=image_size
        ),
        
        # Normalización
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # Para 6 canales (3 pre + 3 post)
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        
        # Convertir a tensor
        ToTensorV2()
    ])


def get_val_test_transforms(image_size: int) -> A.Compose:
    """
    Obtener transformaciones para validación y test (sin aumentación).
    
    Args:
        image_size: Tamaño de las imágenes de salida
        
    Returns:
        Pipeline de transformaciones para validación/test
    """
    
    return A.Compose([
        # Solo redimensionar y normalizar
        A.Resize(
            height=image_size,
            width=image_size
        ),
        
        # Normalización
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # Para 6 canales
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        
        # Convertir a tensor
        ToTensorV2()
    ])


def get_inference_transforms(image_size: int) -> A.Compose:
    """
    Obtener transformaciones para inferencia.
    
    Args:
        image_size: Tamaño de las imágenes de salida
        
    Returns:
        Pipeline de transformaciones para inferencia
    """
    
    return A.Compose([
        # Redimensionar manteniendo aspect ratio
        A.Resize(
            height=image_size,
            width=image_size
        ),
        
        # Normalización
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        
        # Convertir a tensor
        ToTensorV2()
    ])


def get_tta_transforms(image_size: int) -> A.Compose:
    """
    Obtener transformaciones para Test Time Augmentation (TTA).
    
    Args:
        image_size: Tamaño de las imágenes de salida
        
    Returns:
        Pipeline de transformaciones para TTA
    """
    
    return A.Compose([
        # Transformaciones básicas para TTA
        A.RandomRotate90(p=1.0),
        A.Flip(p=0.5),
        
        # Redimensionar
        A.Resize(
            height=image_size,
            width=image_size
        ),
        
        # Normalización
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        
        # Convertir a tensor
        ToTensorV2()
    ])


def get_strong_augmentation_transforms(image_size: int) -> A.Compose:
    """
    Obtener transformaciones de aumentación fuerte para casos especiales.
    
    Args:
        image_size: Tamaño de las imágenes de salida
        
    Returns:
        Pipeline de transformaciones con aumentación fuerte
    """
    
    return A.Compose([
        # Transformaciones geométricas más agresivas
        A.RandomRotate90(p=0.8),
        A.Flip(p=0.8),
        A.Transpose(p=0.8),
        
        # Transformaciones de escala más agresivas
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.8
        ),
        
        # Transformaciones de perspectiva más agresivas
        A.Perspective(
            scale=(0.1, 0.2),
            p=0.5
        ),
        
        # Transformaciones de color más agresivas
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=1.0
            ),
        ], p=0.8),
        
        A.HueSaturationValue(
            hue_shift_limit=30,
            sat_shift_limit=40,
            val_shift_limit=30,
            p=0.8
        ),
        
        # Transformaciones de ruido más agresivas
        A.OneOf([
            A.GaussNoise(
                var_limit=(20.0, 80.0),
                p=1.0
            ),
            A.ISONoise(
                color_shift=(0.02, 0.08),
                intensity=(0.2, 0.8),
                p=1.0
            ),
        ], p=0.6),
        
        # Transformaciones de blur más agresivas
        A.OneOf([
            A.MotionBlur(
                blur_limit=5,
                p=1.0
            ),
            A.MedianBlur(
                blur_limit=5,
                p=1.0
            ),
            A.Blur(
                blur_limit=5,
                p=1.0
            ),
        ], p=0.6),
        
        # Transformaciones específicas para imágenes satelitales
        A.OneOf([
            A.ElasticTransform(
                alpha=2,
                sigma=100,
                alpha_affine=100,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=10,
                distort_limit=0.5,
                p=1.0
            ),
        ], p=0.5),
        
        # Redimensionar
        A.Resize(
            height=image_size,
            width=image_size
        ),
        
        # Normalización
        A.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
        
        # Convertir a tensor
        ToTensorV2()
    ])


def apply_tta_transforms(
    image: np.ndarray,
    transforms: A.Compose,
    num_transforms: int = 4
) -> list:
    """
    Aplicar Test Time Augmentation a una imagen.
    
    Args:
        image: Imagen de entrada
        transforms: Pipeline de transformaciones
        num_transforms: Número de transformaciones a aplicar
        
    Returns:
        Lista de imágenes transformadas
    """
    
    transformed_images = []
    
    for i in range(num_transforms):
        transformed = transforms(image=image)
        transformed_images.append(transformed['image'])
    
    return transformed_images


def denormalize_image(
    image: np.ndarray,
    mean: list = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Desnormalizar una imagen.
    
    Args:
        image: Imagen normalizada
        mean: Valores medios de normalización
        std: Valores de desviación estándar de normalización
        
    Returns:
        Imagen desnormalizada
    """
    
    mean = np.array(mean)
    std = np.array(std)
    
    # Desnormalizar
    denormalized = image * std + mean
    
    # Clampear a [0, 1]
    denormalized = np.clip(denormalized, 0, 1)
    
    return denormalized 