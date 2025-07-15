Además de lo que ya resumimos, aquí tienes un desglose más concreto de los **datos disponibles**, qué tendrás que **predecir** y cómo están estructurados:

---

## 📦 ¿Qué incluye el dataset?

1. **Imágenes Satelitales**

   * **Pre-evento** y **post-evento** (ópticas multiespectrales, a veces con banda pancromática asociada)
   * Resolución espacial típica: 0.5 – 1 m/píxel (Pleiades)
   * Formato: GeoTIFF con georreferenciación WGS84
   * Algunas escenas ya vienen pan-sharpened; otras incluyen la banda pancromática por separado.

2. **Polígonos de Edificios**

   * Extraídos de OpenStreetMap o generados manualmente
   * Formato: Shapefile (.shp) o GeoJSON
   * Cada polígono tiene un atributo `damage` (donde está disponible):

     * **0** = indemne
     * **1** = dañado

3. **Anotaciones de Nivel Objeto**

   * Para un subconjunto de edificios: etiqueta binaria (dañado vs. indemne).
   * No cubren todos los edificios de la escena, pero sí muestras representativas de zonas afectadas y no afectadas.

---

## 🎯 ¿Qué hay que predecir?

### 1. Clasificación de cada edificio

* **Tarea principal**: dado el recorte de imagen (post-evento, o pre+post), asignar a cada polígono un label `{0,1}`.
* Métricas habituales:

  * **IoU** (Intersection over Union) sobre polígono vs. máscara de verdad tierra
  * **F1-score** / **Precision-Recall** en la detección de edificios dañados
  * **Accuracy** por edificio (aunque el dataset está desbalanceado hacia no dañados)

### 2. (Opcional) Segmentación Semántica

* Predecir una **máscara raster** de “daño” dentro de cada tile, útil si quieres refinar el contorno del daño.
* Métrica: IoU / Dice sobre píxeles etiquetados.

---

## 🔧 Cómo están organizados los ficheros

```
└─ dataset/
   ├─ pre_event/            # GeoTIFFs multiespectrales o pancromáticos
   ├─ post_event/           # GeoTIFFs multiespectrales o pancromáticos
   ├─ pan/                  # (si aplica) bandas pancromáticas
   ├─ polygons/
   │    ├─ all_buildings.shp   # shapefile con todos los polígonos
   │    └─ labels.geojson      # subset anotado con campo “damage”
   └─ splits/
        ├─ train.txt       # lista de IDs de escena para entrenamiento
        ├─ val.txt         # validación
        └─ test.txt        # evaluación final (sin labels entregados)
```

* Cada escena viene con un **ID único** que empareja

  * el par pre/post-evento
  * el fichero de polígonos

---

## 🛠 Flujo típico de pre-procesamiento

1. **Pan-sharpening** (si tienes banda pancromática)
2. **Co-registro** de pre vs. post (alineación geométrica)
3. **Recorte** de tiles (patches) centrados en cada polígono, con algo de “padding”
4. **Normalización** de bandas (min-max o z-score)
5. **Aumentos** (rotaciones, flips, cambios de brillo/contraste)
6. **Vector → Raster**: rasterizar el polígono para obtener la máscara de verdad tierra

---

Con todo esto claro, el **target** de tu modelo es asignar a cada edificio la etiqueta “dañado” o “indemne”, aprovechando la información espacial y espectral que ofrecen las imágenes multitemporales.


-----


| Paso                      | Herramienta                           | Salida                                |
| ------------------------- | ------------------------------------- | ------------------------------------- |
| 1. Pan-sharpen            | `rasterio` + `rio-tiler` o `gdal`     | COG pancromático fusionado            |
| 2. Co-registro            | `rasterio.warp`, `opencv` o `elastix` | COG pre‐evento alineado a post‐evento |
| 3. Tileado                | `rio-tiler`, `rasterio.windows`       | Parches multiespectrales COG          |
| 4. Normalización          | NumPy / PyTorch transforms            | Parches normalizados (float32)        |
| 5. Rasterizado de máscara | `geopandas` + `rasterio.features`     | Máscaras binarias en COG o Zarr       |


preprocessed/
├─ scenes/              # GeoTIFFs o Zarr por escena
│   ├─ 0001_pre_reg.COG
│   ├─ 0001_post_reg.COG
│   └─ …  
├─ tiles/               # Parches listos para entrenamiento
│   ├─ 0001/             # folder por escena
│   │   ├─ img_0001_001.npy
│   │   ├─ mask_0001_001.npy
│   │   └─ meta_0001_001.json
│   └─ …
└─ index.parquet        # Metadatos de cada parche
