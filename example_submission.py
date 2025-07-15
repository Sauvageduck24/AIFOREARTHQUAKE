import os
import shutil
import random
import geopandas as gpd
import numpy as np

# Ruta al ejemplo de submission para extraer estructura y formato
EXAMPLE_DIR = 'data/international-charter-eo4ai-etq-challenge/example_submission_unzipped/example_submission'
SUBMISSIONS_DIR = 'submissions'

# Crear carpeta submissions si no existe
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Buscar el siguiente número de submission disponible
def next_submission_index(base_dir):
    existing = [d for d in os.listdir(base_dir) if d.startswith('submission_') and os.path.isdir(os.path.join(base_dir, d))]
    nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    return max(nums, default=0) + 1

idx = next_submission_index(SUBMISSIONS_DIR)
submission_folder = f'submission_{idx}'
submission_path = os.path.join(SUBMISSIONS_DIR, submission_folder)
os.makedirs(submission_path, exist_ok=True)

# Listar archivos de ejemplo para replicar nombres y estructura
eg_files = [f for f in os.listdir(EXAMPLE_DIR) if f.endswith('.gpkg')]

for eg_file in eg_files:
    # Leer el archivo de ejemplo para extraer columnas, crs y estructura
    eg_path = os.path.join(EXAMPLE_DIR, eg_file)
    gdf = gpd.read_file(eg_path)
    
    # --- Aquí se usa la estructura y formato del ejemplo ---
    columns = list(gdf.columns)
    crs = gdf.crs
    
    # Crear datos aleatorios: mismo número de filas, mismas columnas menos 'damage'
    n = len(gdf)
    base_cols = [col for col in columns if col != 'damage']
    data = {col: gdf[col] for col in base_cols}
    data['damage'] = np.random.randint(0, 2, size=n)
    data['geometry'] = gdf['geometry']
    out_gdf = gpd.GeoDataFrame(data, crs=crs)
    out_path = os.path.join(submission_path, eg_file.replace('_Ex', ''))
    out_gdf.to_file(out_path, driver='GPKG')

# Comprimir la carpeta de submission en un zip dentro de submissions
zip_base = os.path.join(SUBMISSIONS_DIR, submission_folder)
shutil.make_archive(zip_base, 'zip', submission_path)

print(f"Carpeta y zip generados: {submission_path}, {zip_base}.zip") 