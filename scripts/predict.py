#!/usr/bin/env python3
"""
Script de predicción para AI for Earthquake Response.

Este script realiza predicciones en nuevos datos usando modelos entrenados.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
import rasterio
from loguru import logger

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config, get_default_config
from src.model import get_model
from src.transforms import get_inference_transforms
from src.utils import set_seed, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predicción de daños por terremotos en nuevos datos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Predicción básica
  python scripts/predict.py --model_path outputs/models/best_model.pth --input_path data/new_scene

  # Predicción con configuración específica
  python scripts/predict.py --model_path outputs/models/best_model.pth --input_path data/new_scene --config configs/unet_resnet50.yaml

  # Predicción con umbral personalizado
  python scripts/predict.py --model_path outputs/models/best_model.pth --input_path data/new_scene --threshold 0.7

  # Predicción con TTA
  python scripts/predict.py --model_path outputs/models/best_model.pth --input_path data/new_scene --use_tta

  # Predicción en archivo individual
  python scripts/predict.py --model_path outputs/models/best_model.pth --input_path data/new_scene/image.tif --output_path predictions/
        """
    )
    
    # Parámetros obligatorios
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al modelo entrenado")
    parser.add_argument("--input_path", type=str, required=True, help="Ruta a los datos de entrada")
    
    # Configuración
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML")
    parser.add_argument("--output_path", type=str, help="Directorio de salida para predicciones")
    
    # Parámetros de predicción
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para binarización")
    parser.add_argument("--use_tta", action="store_true", help="Usar Test Time Augmentation")
    parser.add_argument("--save_probabilities", action="store_true", help="Guardar probabilidades además de predicciones binarias")
    
    # Parámetros de hardware
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], help="Dispositivo a usar")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamaño de batch para predicción")
    
    # Parámetros de logging
    parser.add_argument("--verbose", action="store_true", help="Modo verbose")
    
    return parser.parse_args()


def load_image_pair(pre_path, post_path, image_size=512):
    """
    Cargar par de imágenes pre y post terremoto.
    
    Args:
        pre_path: Ruta a imagen pre-terremoto
        post_path: Ruta a imagen post-terremoto
        image_size: Tamaño de imagen de salida
        
    Returns:
        Tensor con imágenes concatenadas
    """
    try:
        # Cargar imagen pre-terremoto
        if pre_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(pre_path) as src:
                pre_image = src.read()
                pre_image = np.transpose(pre_image, (1, 2, 0))
        else:
            pre_image = np.array(Image.open(pre_path))
        
        # Cargar imagen post-terremoto
        if post_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(post_path) as src:
                post_image = src.read()
                post_image = np.transpose(post_image, (1, 2, 0))
        else:
            post_image = np.array(Image.open(post_path))
        
        # Normalizar
        if pre_image.dtype == np.uint16:
            pre_image = pre_image.astype(np.float32) / 65535.0
        elif pre_image.dtype == np.uint8:
            pre_image = pre_image.astype(np.float32) / 255.0
            
        if post_image.dtype == np.uint16:
            post_image = post_image.astype(np.float32) / 65535.0
        elif post_image.dtype == np.uint8:
            post_image = post_image.astype(np.float32) / 255.0
        
        # Asegurar formato correcto
        if len(pre_image.shape) == 2:
            pre_image = np.stack([pre_image] * 3, axis=2)
        elif pre_image.shape[2] > 3:
            pre_image = pre_image[:, :, :3]
            
        if len(post_image.shape) == 2:
            post_image = np.stack([post_image] * 3, axis=2)
        elif post_image.shape[2] > 3:
            post_image = post_image[:, :, :3]
        
        # Concatenar imágenes
        combined_image = np.concatenate([pre_image, post_image], axis=2)
        
        return combined_image
        
    except Exception as e:
        logger.error(f"Error cargando imágenes: {e}")
        return None


def predict_single_image(model, image, transforms, device, threshold=0.5, use_tta=False):
    """
    Realizar predicción en una imagen.
    
    Args:
        model: Modelo entrenado
        image: Imagen de entrada
        transforms: Transformaciones a aplicar
        device: Dispositivo a usar
        threshold: Umbral para binarización
        use_tta: Si usar Test Time Augmentation
        
    Returns:
        Predicción binaria y probabilidades
    """
    model.eval()
    
    with torch.no_grad():
        if use_tta:
            # Test Time Augmentation
            predictions = []
            
            # Transformaciones para TTA
            tta_transforms = [
                lambda x: x,  # Original
                lambda x: np.flip(x, axis=1),  # Flip horizontal
                lambda x: np.flip(x, axis=0),  # Flip vertical
                lambda x: np.rot90(x, k=2)  # Rotate 180
            ]
            
            for transform in tta_transforms:
                # Aplicar transformación
                transformed_image = transform(image)
                
                # Aplicar transformaciones de preprocesamiento
                transformed = transforms(image=transformed_image)
                input_tensor = transformed['image'].unsqueeze(0).to(device)
                
                # Predicción
                pred = model(input_tensor)
                pred = torch.sigmoid(pred)
                
                # Revertir transformación
                pred = pred.cpu().numpy().squeeze()
                if transform != tta_transforms[0]:  # Si no es la original
                    if transform == tta_transforms[1]:  # Flip horizontal
                        pred = np.flip(pred, axis=1)
                    elif transform == tta_transforms[2]:  # Flip vertical
                        pred = np.flip(pred, axis=0)
                    elif transform == tta_transforms[3]:  # Rotate 180
                        pred = np.rot90(pred, k=2)
                
                predictions.append(pred)
            
            # Promedio de predicciones
            probabilities = np.mean(predictions, axis=0)
        else:
            # Predicción simple
            transformed = transforms(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            pred = model(input_tensor)
            probabilities = torch.sigmoid(pred).cpu().numpy().squeeze()
        
        # Binarizar predicción
        binary_prediction = (probabilities > threshold).astype(np.uint8)
        
        return binary_prediction, probabilities


def predict_directory(model, input_dir, output_dir, config, device, threshold=0.5, use_tta=False, save_probabilities=False):
    """
    Realizar predicciones en un directorio de datos.
    
    Args:
        model: Modelo entrenado
        input_dir: Directorio de entrada
        output_dir: Directorio de salida
        config: Configuración del experimento
        device: Dispositivo a usar
        threshold: Umbral para binarización
        use_tta: Si usar Test Time Augmentation
        save_probabilities: Si guardar probabilidades
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Obtener transformaciones
    transforms = get_inference_transforms(config.data.image_size)
    
    # Buscar pares de imágenes
    scene_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('scene_')]
    
    if not scene_dirs:
        logger.warning(f"No se encontraron directorios de escenas en {input_dir}")
        return
    
    logger.info(f"Encontradas {len(scene_dirs)} escenas para predicción")
    
    for scene_dir in scene_dirs:
        logger.info(f"Procesando escena: {scene_dir.name}")
        
        # Buscar imágenes pre y post
        pre_event_dir = scene_dir / "pre_event"
        post_event_dir = scene_dir / "post_event"
        
        if not (pre_event_dir.exists() and post_event_dir.exists()):
            logger.warning(f"Estructura incompleta en {scene_dir.name}, saltando...")
            continue
        
        # Encontrar archivos de imagen
        pre_images = list(pre_event_dir.glob("*.tif")) + list(pre_event_dir.glob("*.tiff")) + list(pre_event_dir.glob("*.jpg")) + list(pre_event_dir.glob("*.png"))
        post_images = list(post_event_dir.glob("*.tif")) + list(post_event_dir.glob("*.tiff")) + list(post_event_dir.glob("*.jpg")) + list(post_event_dir.glob("*.png"))
        
        if not pre_images or not post_images:
            logger.warning(f"No se encontraron imágenes en {scene_dir.name}, saltando...")
            continue
        
        # Crear directorio de salida para la escena
        scene_output_dir = output_path / scene_dir.name
        scene_output_dir.mkdir(exist_ok=True)
        
        # Procesar cada par de imágenes
        for pre_img, post_img in zip(pre_images, post_images):
            logger.info(f"  Procesando: {pre_img.name} - {post_img.name}")
            
            # Cargar imágenes
            combined_image = load_image_pair(str(pre_img), str(post_img), config.data.image_size)
            
            if combined_image is None:
                logger.warning(f"Error cargando imágenes, saltando...")
                continue
            
            # Realizar predicción
            binary_pred, probabilities = predict_single_image(
                model=model,
                image=combined_image,
                transforms=transforms,
                device=device,
                threshold=threshold,
                use_tta=use_tta
            )
            
            # Guardar predicciones
            base_name = pre_img.stem.replace("_pre", "").replace("pre_", "")
            
            # Guardar predicción binaria
            binary_path = scene_output_dir / f"{base_name}_damage_mask.png"
            cv2.imwrite(str(binary_path), binary_pred * 255)
            
            # Guardar probabilidades si se solicita
            if save_probabilities:
                prob_path = scene_output_dir / f"{base_name}_damage_probabilities.tif"
                with rasterio.open(
                    str(prob_path),
                    'w',
                    driver='GTiff',
                    height=probabilities.shape[0],
                    width=probabilities.shape[1],
                    count=1,
                    dtype=probabilities.dtype
                ) as dst:
                    dst.write(probabilities, 1)
            
            logger.info(f"    Predicción guardada: {binary_path}")
    
    logger.info(f"Predicciones completadas. Resultados en: {output_path}")


def predict_single_file(model, input_file, output_dir, config, device, threshold=0.5, use_tta=False, save_probabilities=False):
    """
    Realizar predicción en un archivo individual.
    
    Args:
        model: Modelo entrenado
        input_file: Archivo de entrada
        output_dir: Directorio de salida
        config: Configuración del experimento
        device: Dispositivo a usar
        threshold: Umbral para binarización
        use_tta: Si usar Test Time Augmentation
        save_probabilities: Si guardar probabilidades
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Obtener transformaciones
    transforms = get_inference_transforms(config.data.image_size)
    
    # Cargar imagen (asumiendo que ya está concatenada)
    try:
        if input_path.suffix.lower() in ['.tif', '.tiff']:
            with rasterio.open(input_path) as src:
                image = src.read()
                image = np.transpose(image, (1, 2, 0))
        else:
            image = np.array(Image.open(input_path))
        
        # Normalizar
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        logger.info(f"Imagen cargada: {image.shape}")
        
    except Exception as e:
        logger.error(f"Error cargando imagen: {e}")
        return
    
    # Realizar predicción
    binary_pred, probabilities = predict_single_image(
        model=model,
        image=image,
        transforms=transforms,
        device=device,
        threshold=threshold,
        use_tta=use_tta
    )
    
    # Guardar predicciones
    base_name = input_path.stem
    
    # Guardar predicción binaria
    binary_path = output_path / f"{base_name}_damage_mask.png"
    cv2.imwrite(str(binary_path), binary_pred * 255)
    
    # Guardar probabilidades si se solicita
    if save_probabilities:
        prob_path = output_path / f"{base_name}_damage_probabilities.tif"
        with rasterio.open(
            str(prob_path),
            'w',
            driver='GTiff',
            height=probabilities.shape[0],
            width=probabilities.shape[1],
            count=1,
            dtype=probabilities.dtype
        ) as dst:
            dst.write(probabilities, 1)
    
    logger.info(f"Predicción guardada: {binary_path}")
    if save_probabilities:
        logger.info(f"Probabilidades guardadas: {prob_path}")


def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar logging
    setup_logging()
    
    # Cargar configuración
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Sobrescribir configuración
    if args.output_path:
        config.logging.output_dir = args.output_path
    if args.device:
        config.hardware.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # Configurar semilla
    set_seed(config.seed, config.deterministic)
    
    # Configurar dispositivo
    device = torch.device(config.hardware.device)
    logger.info(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    logger.info(f"Cargando modelo desde: {args.model_path}")
    model = get_model(config.model)
    model = model.to(device)
    
    # Cargar checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Modelo cargado desde checkpoint (época {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Modelo cargado (solo pesos)")
    
    # Determinar tipo de entrada
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Predicción en archivo individual
        logger.info(f"Predicción en archivo individual: {input_path}")
        predict_single_file(
            model=model,
            input_file=args.input_path,
            output_dir=args.output_path or config.logging.output_dir,
            config=config,
            device=device,
            threshold=args.threshold,
            use_tta=args.use_tta,
            save_probabilities=args.save_probabilities
        )
    elif input_path.is_dir():
        # Predicción en directorio
        logger.info(f"Predicción en directorio: {input_path}")
        predict_directory(
            model=model,
            input_dir=args.input_path,
            output_dir=args.output_path or config.logging.output_dir,
            config=config,
            device=device,
            threshold=args.threshold,
            use_tta=args.use_tta,
            save_probabilities=args.save_probabilities
        )
    else:
        logger.error(f"La ruta {args.input_path} no existe")
        return
    
    logger.info("Predicción completada exitosamente")


if __name__ == "__main__":
    main() 