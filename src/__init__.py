"""
AI for Earthquake Response - Deep Learning Project

Este módulo contiene el código fuente principal para la detección
automática de daños en edificios tras terremotos usando imágenes satelitales.
"""

__version__ = "1.0.0"
__author__ = "AI4EO Competition Team"
__email__ = "contact@ai4eo.eu"

from config import Config
# from data_loader import EarthquakeDataset, get_data_loaders
# Si necesitas get_data_loaders, descomenta la siguiente línea:
# from data_loader import get_data_loaders
from model import get_model
from transforms import get_transforms
from utils import setup_logging, calculate_metrics

__all__ = [
    "Config",
    "get_model",
    "get_transforms",
    "setup_logging",
    "calculate_metrics"
] 