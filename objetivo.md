## Proyecto AI for Earthquake Response

Este documento describe de forma resumida y organizada los **objetivos**, **datos**, **preprocesamiento** y **estrategia de modelado** para la competición AI for Earthquake Response.

---

### 1. Objetivo del proyecto

Desarrollar una **pipeline de IA** que, dadas imágenes satelitales pre- y post-terremoto y un archivo de polígonos de edificios, asigne una etiqueta de **dañado (1)** o **indemne (0)** a cada edificio. El fin es acelerar la generación de mapas de daños tras un sismo.

---

### 2. Descripción del dataset

#### 2.1 Imágenes Satelitales

* **Tipos**: ópticas multiespectrales, algunas con banda pancromática.
* **Temporalidad**: escenas **pre-evento** y **post-evento**.
* **Resolución**: \~0.5 – 1 m/píxel (Pleiades).
* **Formato**: GeoTIFF georreferenciado (WGS84).
* **Pan-sharpening**: algunos archivos ya preprocesados, otros con bandas pancromáticas separadas.

#### 2.2 Polígonos de Edificios

* **Origen**: OpenStreetMap o anotaciones manuales.
* **Formato**: Shapefile (.shp) o GeoJSON.
* **Atributos**:

  * `building_id`: identificador único.
  * `damage` (cuando existe): 0 (indemne), 1 (dañado).

#### 2.3 Anotaciones de Nivel Objeto

* Etiquetas binarias (dañado vs. indemne) para un **subconjunto** de edificios.
* No cubren todos los edificios, pero ofrecen muestras representativas.

---

### 3. Tareas de predicción

1. **Clasificación de edificios**:

   * Input: recortes de imagen (post-evento o pre+post).
   * Output: etiqueta `{0,1}` por edificio.
   * Métricas: IoU, F1-score, Precision–Recall, Accuracy.

2. **(Opcional) Segmentación semántica**:

   * Input: tile completo.
   * Output: máscara raster de pixels dañados.
   * Métricas: IoU o Dice.

---

### 4. Organización de ficheros real (data/international-charter-eo4ai-etq-challenge)

```
data/international-charter-eo4ai-etq-challenge/
├─ Training/
│   ├─ Call_1075_China/
│   │   ├─ WV03N28_549166E087_4023612023062600000000PN00-calibrated/
│   │   │   ├─ Optical_Calibration/
│   │   │   │   ├─ overview-pan.tif
│   │   │   │   └─ r-pan.tif
│   │   │   └─ WV03N28_549166E087_4023612023062600000000PN00-calibrated.json
│   │   └─ ...
│   ├─ ...
│   ├─ Annotations/
│   │   ├─ Call_1075_China/
│   │   │   ├─ china_training.gpkg
│   │   │   └─ china_testing.gpkg
│   │   ├─ Call_919_Turkey/
│   │   │   ├─ antakya_east_training.gpkg
│   │   │   ├─ antakya_east_testing.gpkg
│   │   │   ├─ marash_training.gpkg
│   │   │   ├─ marash_testing.gpkg
│   │   │   ├─ adiyaman_training.gpkg
│   │   │   ├─ adiyaman_testing.gpkg
│   │   │   └─ ...
│   │   └─ ...
│   └─ training_dataset.contents
├─ example_submission.zip
├─ example_submission_unzipped/
│   └─ example_submission/
│       ├─ china_testing_Ex.gpkg
│       ├─ marash_testing_Ex.gpkg
│       ├─ adiyaman_testing_Ex.gpkg
│       └─ antakya_east_testing_Ex.gpkg
├─ Phase_1_Dataset_Details.xlsx
├─ catalog.json
├─ charter-dataset.structure
├─ charter-dataset.filelist
├─ README.md
└─ catalog.v1.parquet
```

---

### 5. Pipeline de preprocesamiento

Un flujo reproducible y paralelo para transformar 200 GB de datos en parches listos para entrenamiento:

| Paso                    | Herramienta                      | Output                          |
| ----------------------- | -------------------------------- | ------------------------------- |
| 1. Pan-sharpen          | rasterio / GDAL                  | COG pancromático fusionado      |
| 2. Co-registro          | rasterio.warp / OpenCV / elastix | COG pre-evento alineado         |
| 3. Tileado              | rio-tiler / rasterio.windows     | Parches multiespectrales COG    |
| 4. Normalización        | NumPy / PyTorch transforms       | Parches float32 normalizados    |
| 5. Rasterizado máscaras | geopandas + rasterio.features    | Máscaras binarias en COG o Zarr |

#### 5.1 Formato de salida

```
preprocessed/
├─ scenes/                # GeoTIFFs o Zarr por escena
│   ├─ 0001_pre_reg.COG
│   ├─ 0001_post_reg.COG
│   └─ …
├─ tiles/                 # Parches para entrenamiento
│   ├─ 0001/
│   │   ├─ img_0001_001.npy
│   │   ├─ mask_0001_001.npy
│   │   └─ meta_0001_001.json
│   └─ …
└─ index.parquet          # Metadatos por parche
```

#### 5.2 Orquestación y versionado

* **Snakemake/Nextflow** para cadenas de trabajo automatizadas.
* **DVC / Git-LFS** para versionado de datos grandes.
* **Conda / Docker** para entornos reproducibles.

---

### 6. Estrategia de modelado: arquitectura multi-cabeza

Combina **generalización global** y **especialización local**:

#### 6.1 Estructura

1. **Backbone compartido** (ResNet/EfficientNet)
2. **Cuatro heads**: una por ciudad, con clasificación binaria
3. **Pérdidas independientes**: BCE o focal loss por head, combinadas en la loss total

#### 6.2 Flujo de entrenamiento

1. Lectura de batch con imágenes y etiqueta de ciudad.
2. Forward por backbone → enrute a head según ciudad.
3. Cálculo de pérdidas \$L\_i\$ y pérdida agregada \$L=\frac{1}{4}\sum\_i L\_i\$.
4. Backpropagation conjunta.

#### 6.3 Opcional: adaptación de dominio

* Clasificador adversarial para forzar invariancia de dominio
* Backbone entrenado a confundir dicho clasificador

#### 6.4 Puntos clave

* Balanceo de batches por ciudad
* LR diferenciado: bajo para backbone, alto para heads
* Monitorización F1 por ciudad

#### 6.5 Visualización y monitoreo con Weights & Biases (wandb)

Para el seguimiento y visualización de experimentos, métricas y resultados, se utiliza la plataforma [Weights & Biases](https://wandb.ai/):

- Registro automático de métricas de entrenamiento y validación (loss, F1, accuracy, etc.).
- Visualización de curvas de aprendizaje y comparación entre experimentos.
- Almacenamiento de artefactos como modelos, predicciones y gráficos de resultados.
- Integración sencilla con PyTorch y otros frameworks mediante la API de wandb.

Esto permite un análisis reproducible y colaborativo del desempeño del modelo a lo largo del desarrollo.

---

## Resumen ejecutivo

Este proyecto aborda el desafío AI for Earthquake Response, cuyo objetivo es desarrollar una pipeline de inteligencia artificial capaz de identificar automáticamente edificios dañados tras un terremoto, a partir de imágenes satelitales y datos vectoriales de edificios. El enfoque combina procesamiento geoespacial, aprendizaje profundo y buenas prácticas de experimentación para acelerar la generación de mapas de daños y apoyar la respuesta humanitaria.

---

### Aclaración sobre poligonización

La poligonización de edificios (generación de polígonos vectoriales que representan la huella de cada edificio) **no es parte del challenge**. Se asume que los polígonos de edificios ya están provistos como insumo, y la tarea consiste únicamente en clasificar el daño (0: indemne, 1: dañado) para cada polígono.

---

### Formato y convención de nombres de example_submission.zip

El archivo `example_submission.zip` contiene la estructura y formato esperados para la submission del challenge:

- Dentro del zip hay una carpeta (por ejemplo, `example_submission/`) con varios archivos `.gpkg`, uno por sitio o área de test.
- Cada archivo `.gpkg` debe tener:
  - Una columna `building_id` (identificador único del edificio).
  - Una columna `damage` (0: indemne, 1: dañado).
  - La geometría de cada edificio (polígono).
- El nombre de cada archivo sigue la convención `{site_id}_testing.gpkg` (por ejemplo, `china_testing.gpkg`, `marash_testing.gpkg`).
- El formato de columnas, tipos y sistema de referencia espacial debe ser idéntico al de los archivos de ejemplo.

La submission final debe replicar exactamente esta estructura y convención de nombres para ser válida.

---