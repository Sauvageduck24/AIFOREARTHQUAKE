# AI for Earthquake Response - Deep Learning Project

Este proyecto implementa modelos de deep learning para la detección automática de daños en edificios tras terremotos utilizando imágenes satelitales multitemporales (antes y después del evento).

---

## 🏆 Competición

Este proyecto participa en la competición **[AI for Earthquake Response](https://platform.ai4eo.eu/ai-for-earthquake-response)** organizada por ESA Φ-lab y el International Charter 'Space and Major Disasters'.

## 🎯 Objetivo

Desarrollar modelos de IA capaces de identificar automáticamente edificios dañados vs. no dañados analizando imágenes satelitales de alta resolución pre y post-terremoto.

## 📁 Estructura del Proyecto

La siguiente imagen muestra la estructura general del repositorio:

![Estructura del proyecto](data/imgs/estructura.jpg)

También se detalla a continuación:

```
AIFOREARTHQUAKE/
├── data/                   # Datasets locales y preprocesados
│   └── imgs/              # Imágenes de documentación
├── configs/               # Configuraciones de experimentos
├── src/                    # Código fuente principal
│   ├── data_loader.py      # Carga de datos y DataLoaders
│   ├── transforms.py       # Transformaciones y aumentos
│   ├── model.py            # Definición de modelos
│   ├── train.py            # Entrenamiento y validación
│   ├── utils.py            # Utilidades y métricas
│   ├── config.py           # Configuración del experimento
│   └── preprocess.py       # Preprocesamiento de datos
├── outputs/                # Modelos, logs y predicciones
├── preprocessed/           # Datos preprocesados
├── proof_scripts/          # Scripts de prueba y análisis
├── requirements.txt        # Dependencias
└── README.md               # Este archivo
```

## 📝 Descripción de scripts principales

### `src/preprocess.py`
- **Función:** Preprocesa los datos crudos, generando tiles (parches) de imágenes pre y post-terremoto y sus máscaras de daño asociadas. Organiza los datos en splits (`full`, `test`, etc.) y guarda metadatos.
- **Uso típico:**
  ```bash
  python src/preprocess.py --raw_data_root data/raw --output_root preprocessed --binary --seed 42
  ```
- **Notas:** Permite configurar tamaño de tile, solapamiento, filtrado, y reproducibilidad. Los datos resultantes se usan para entrenamiento y evaluación.

### `src/data_loader.py`
- **Función:** Carga los datos preprocesados y crea objetos `Dataset` y `DataLoader` de PyTorch, listos para entrenamiento, validación y test. Permite balanceo de clases y augmentations.
- **Uso típico:**
  ```bash
  python src/data_loader.py --data_root preprocessed --mode pre_post --filter_has_data --batch_size 8
  ```
- **Notas:** Permite elegir modo de datos (`pre_post`, `pre_only`, `post_only`), filtrar tiles sin datos, y guardar los datasets generados. Imprime información útil sobre los datos cargados.

### `src/train.py`
- **Función:** Entrena el modelo de segmentación de daño usando los dataloaders generados. Permite configurar arquitectura, hiperparámetros, augmentations, logging, y checkpointing.
- **Uso típico:**
  ```bash
  python src/train.py --config configs/baseline.yaml
  ```
- **Notas:** Lee la configuración desde un archivo YAML, entrena el modelo, guarda los mejores pesos y logs de entrenamiento.

### `src/test.py`
- **Función:** Evalúa el modelo entrenado sobre el split de test, generando métricas y predicciones. Permite cargar modelos y dataloaders guardados.
- **Uso típico:**
  ```bash
  python src/test.py --config configs/baseline.yaml --model_path outputs/models/best_model.pth
  ```
- **Notas:** Calcula métricas como IoU, Dice, precisión y recall sobre el conjunto de test.

## 🚀 Instalación

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd AIFOREARTHQUAKE
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 📊 Preparación de Datos

1. **Acceso al dataset:**
   - Registrarse en [AI4EO Platform](https://platform.ai4eo.eu/ai-for-earthquake-response)
   - Crear cuenta en [EOTDL](https://www.eotdl.com) con el mismo email
   - Esperar hasta 48h para acceso al dataset

2. **Estructura de datos esperada:**
```
data/
├── raw/                    # Datos originales
│   ├── scene_001/
│   │   ├── pre_event/      # Imágenes pre-terremoto
│   │   ├── post_event/     # Imágenes post-terremoto
│   │   └── annotations/    # Máscaras de daño
│   └── ...
├── processed/              # Datos preprocesados
└── splits/                 # Particiones train/val/test
```

### Detalles técnicos del preprocesamiento (`src/preprocess.py`)

El script de preprocesamiento divide las imágenes satelitales en pequeños parches (tiles) y genera las máscaras de daño asociadas, listas para el entrenamiento. A continuación se detallan los aspectos clave:

- **Tiling configurable:**
  - `TILE_SIZE`: Tamaño de cada tile cuadrado (por defecto 512x512 píxeles).
  - `TILE_OVERLAP`: Solapamiento entre tiles adyacentes (por defecto 64 píxeles), lo que permite que los objetos en los bordes aparezcan completos en varios tiles.
  - El tiling recorre la imagen en pasos de `TILE_SIZE - TILE_OVERLAP`.

- **Filtrado de tiles:**
  - `MIN_VALID_PIXELS`: Porcentaje mínimo de píxeles válidos (no fondo) requerido para guardar un tile. Tiles con menos proporción de píxeles útiles se descartan automáticamente.

- **Máscaras y clases:**
  - La máscara rasterizada se genera a partir de la columna `damaged` de las anotaciones vectoriales.
  - Si se usa el flag `--binary`, la máscara es binaria: 0 = no dañado, 1 = dañado, 255 = desconocido (NaN, None, valores inesperados).
  - Los tiles que contienen algún valor 255 (desconocido) en la máscara se asignan al split `test` y no se usan para entrenamiento.
  - Si la máscara solo contiene 0 y 1, el tile se asigna a `full`/`small`.

- **Reproducibilidad:**
  - El script acepta el argumento `--seed` para fijar la semilla de random, numpy y torch, garantizando splits y resultados reproducibles.

- **Formato de guardado eficiente:**
  - Los tiles de imagen y máscara se guardan como archivos `.npz` comprimidos (usando `np.savez_compressed`), lo que reduce el uso de disco 2-3 veces respecto a formatos tradicionales.
  - Los metadatos de cada tile (incluyendo paths, fechas, bandas, etc.) se guardan en archivos `meta.json` por split.

- **Ejemplo de uso:**

```bash
python src/preprocess.py \
  --raw_data_root data/international-charter-eo4ai-etq-challenge/Training \
  --output_root data/preprocessed \
  --binary \
  --seed 42 \
  --workers 4
```

- **Advertencia:**
  - El preprocesamiento puede requerir bastante espacio en disco y memoria RAM, especialmente si se procesan todas las ciudades y se usan tiles grandes o poco solapamiento.
  - Se recomienda ejecutar primero en modo prueba (`--prueba`) para validar la configuración.

Para más detalles, consulta el código fuente de `src/preprocess.py` y los ejemplos de salida en la carpeta `data/preprocessed/`.

## ✅ Revisión de alineación entre preprocess.py y el dataset

La siguiente tabla resume cómo el script `src/preprocess.py` se alinea con los requisitos y recomendaciones del dataset oficial y este repositorio:

| Requisito (README dataset)           | ¿Implementado en preprocess.py? | Notas                                  |
|--------------------------------------|:-------------------------------:|----------------------------------------|
| Estructura de directorios y archivos | ✔️                              | Descubrimiento y emparejamiento robusto|
| Manejo de bandas y pan-sharpening    | ✔️                              | Método GSA, opcional                   |
| Rasterización de anotaciones         | ✔️                              | Daño/no-daño/desconocido               |
| Emparejamiento pre/post evento       | ✔️                              | Fecha más cercana antes/después evento |
| Filtrado de tiles                    | ✔️                              | Basado en cobertura de máscara         |
| Estructura de salida y metadatos     | ✔️                              | .npz, meta.json, stats.json            |
| Manejo de errores                    | ✔️                              | rich console, warnings, skips          |
| Recorte por "analysis extent"        | ❌/⚠️                           | No implementado, considerar agregar    |
| Manejo de desbalance de clases       | ⚠️                              | No abordado, relevante para entrenamiento|
| Augmentación                         | ⚠️                              | No en preprocesado, sí en entrenamiento|

**Sugerencias:**
- Si el dataset incluye polígonos de "analysis extent", considera agregar lógica para recortar/máscarar tiles a esa área.
- Documenta cómo se maneja el desbalance de clases y las anotaciones incompletas.
- El preprocesamiento es robusto y modular, listo para grandes volúmenes de datos.

---

## 📦 Generación de Datasets y DataLoaders para Entrenamiento

Una vez preprocesados los datos, puedes generar los objetos `Dataset` y `DataLoader` de PyTorch listos para entrenar modelos que predicen el daño (`damaged`).

Esto se realiza con el script [`src/data_loader.py`](src/data_loader.py):

- Permite elegir entre el split completo (`full`) o reducido (`small`) usando el argumento `--small`.
- Carga los tiles y máscaras desde los archivos `.npz` y los metadatos de `meta.json`.
- Guarda los objetos `Dataset` y `DataLoader` en disco para su uso directo en los experimentos.

### Ejemplo de uso

```bash
# Para el split reducido (small, recomendado para pruebas rápidas)
python src/data_loader.py --small

# Para el split completo (full, para entrenamiento final)
python src/data_loader.py
```

Esto generará los archivos:
- `data/datasets/dataset_small.pt` y/o `data/datasets/dataset_full.pt`
- `data/dataloaders/dataloader_small.pt` y/o `data/dataloaders/dataloader_full.pt`

Estos archivos pueden ser cargados directamente en tus scripts de entrenamiento para entrenar modelos de segmentación que predicen la máscara de daño (`damaged`).

### Consejos
- Usa el split `small` para depuración y pruebas rápidas.
- Usa el split `full` para el entrenamiento final del modelo.
- Los dataloaders están listos para PyTorch y cargan los tiles y máscaras en memoria de forma eficiente.

Para más detalles, consulta el código fuente de [`src/data_loader.py`](src/data_loader.py).

## Estructura del dataset y preprocesamiento

El siguiente diagrama muestra cómo están organizados los datos antes y después de ejecutar el preprocesamiento:

![Estructura del dataset](data/imgs/dataset.jpg)

### Descripción del flujo de datos

- **Datos crudos (antes de preprocesar):**
  - Carpeta `Training/` con subcarpetas por ciudad (`Call_<ID>_<Ciudad>/`).
  - Cada ciudad contiene:
    - `collection.json`: catálogo STAC de escenas.
    - Archivos `.json` de escenas calibradas (con referencias a las bandas).
    - Archivos `.tif` de bandas espectrales (blue, green, red, nir, pan).
    - Anotaciones vectoriales de daño en `Annotations/<ID>_<Ciudad>/*.gpkg`.

- **Preprocesamiento (`preprocess.py`):**
  - Lee y organiza las imágenes pre y post evento.
  - Genera tiles normalizados y máscaras rasterizadas.
  - Calcula estadísticas globales (media y std).
  - Guarda overlays de debug si se solicita.

- **Datos tras preprocesar (para el dataset):**
  - Carpeta `data/preprocessed/` con splits: `full/`, `small/`, `test/`.
  - Cada split contiene:
    - `images/`: tiles de imágenes como tensores `.pt` ([canales, 512, 512]).
    - `masks/`: máscaras de daño como tensores `.pt` ([512, 512]).
    - `meta.json`: lista de metadatos de cada tile, incluyendo:
      - id, ciudad, escena, bandas, paths, bbox, is_partial, y un campo `images_info` con la información de las imágenes pre y post usadas (scene_id, fecha, tipo).
  - Archivo `stats.json` con media y desviación estándar global.
  - Carpeta `debug_vis/` con visualizaciones opcionales.

Para más detalles, consulta el código fuente de `preprocess.py` y el ejemplo de salida en la carpeta `data/preprocessed/`.

## 🔍 Visualización de Resultados

El proyecto incluye herramientas de visualización para analizar los resultados del modelo:

### Comparación de Escalas
- **Vista cercana:** ![Vista cercana](data/imgs/Figure_1_cerca.png) - Detalle de predicciones a nivel de edificio
- **Vista lejana:** ![Vista lejana](data/imgs/Figure_1_lejos.png) - Vista general del área afectada

Estas visualizaciones permiten evaluar la calidad de las predicciones tanto a nivel local como regional.

## 🏗️ Entrenamiento

### Entrenamiento Básico
```bash
python src/train.py --config configs/baseline.yaml
```

### Entrenamiento Distribuido (DDP)
```bash
python -m torch.distributed.launch --nproc_per_node=4 src/train.py --config configs/baseline.yaml --distributed
```

### Entrenamiento con Múltiples Seeds
```bash
python src/train.py --config configs/baseline.yaml --seed 42
```

## 🔍 Evaluación

```bash
python src/test.py --config configs/baseline.yaml --model_path outputs/models/best_model.pth
```

## 📈 Predicción

```bash
python src/test.py --config configs/baseline.yaml --model_path outputs/models/best_model.pth --predict_only
```

## 📋 Configuración

El proyecto utiliza archivos YAML para la configuración de experimentos:

- **`configs/baseline.yaml`**: Configuración base
- **`configs/unet_resnet50.yaml`**: UNet con ResNet-50
- **`configs/segmentation_transformer.yaml`**: Transformer para segmentación

Cada configuración incluye:
- **Modelo**: Arquitectura, backbone, hiperparámetros
- **Datos**: Rutas, transformaciones, batch size
- **Entrenamiento**: Optimizador, learning rate, epochs
- **Hardware**: GPU, workers, distributed training

## 🧪 Experimentos y Análisis

### Scripts de Prueba y Análisis
El directorio `proof_scripts/` contiene scripts útiles para análisis y depuración:

- `analyze_example_submission.py`: Análisis de la estructura de submission
- `analyze_npz_channels.py`: Análisis de canales en archivos NPZ
- `calc_max_fids_per_tile.py`: Cálculo de FIDs por tile
- `check_band_assets.py`: Verificación de assets de bandas
- `check_tiles.py`: Verificación de tiles generados
- `debug_adiyaman.py`: Depuración específica para Adiyaman
- `evaluate.py`: Evaluación de modelos
- `list_image_dates.py`: Listado de fechas de imágenes
- `test_single_tile.py`: Prueba de un tile individual
- `view_augmented_batch.py`: Visualización de batches aumentados
- `view_gpkg_content.py`: Visualización de contenido GPKG
- `view_prepost_tile.py`: Visualización de tiles pre/post

## 📊 Métricas

El proyecto implementa las siguientes métricas:
- **IoU (Intersection over Union)**
- **Dice Coefficient (F1-Score)**
- **Precision y Recall**
- **Accuracy**

## 🛠️ Características Avanzadas

- ✅ **Entrenamiento Distribuido (DDP)** para escalabilidad
- ✅ **Múltiples Seeds** para reproducibilidad
- ✅ **Logging con TensorBoard/WandB**
- ✅ **Checkpointing** automático
- ✅ **Early Stopping** configurable
- ✅ **Data Augmentation** con Albumentations
- ✅ **Mixed Precision Training** (AMP)
- ✅ **Preprocesamiento eficiente** con compresión NPZ
- ✅ **Visualización avanzada** de resultados

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **ESA Φ-lab** por organizar la competición
- **International Charter 'Space and Major Disasters'** por proporcionar los datos
- **AI4EO Platform** por facilitar la competición

## 📞 Contacto

Para preguntas sobre este proyecto, abrir un issue en GitHub.

---

**Nota**: Este proyecto es parte de la competición [AI for Earthquake Response](https://platform.ai4eo.eu/ai-for-earthquake-response). Los datos están sujetos a restricciones de acceso y deben ser solicitados manualmente al equipo organizador. 

## 👥 Autores

El equipo **MM** está formado por:
- Esteban Sánchez Gámez
- Pablo Delgado Muñoz 