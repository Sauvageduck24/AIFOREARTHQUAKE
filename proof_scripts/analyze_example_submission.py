import os
import geopandas as gpd
import matplotlib.pyplot as plt

EXAMPLE_SUBMISSION_DIR = os.path.join(
    os.path.dirname(__file__),
    '../data/international-charter-eo4ai-etq-challenge/example_submission_unzipped/example_submission'
)

def analyze_gpkg_file(gpkg_path):
    print(f'\n--- Análisis de: {os.path.basename(gpkg_path)} ---')
    gdf = gpd.read_file(gpkg_path)
    print(f'Filas: {len(gdf)}')
    print(f'Columnas: {list(gdf.columns)}')
    if "damaged" in gdf.columns:
        print(f"Valores únicos en 'damaged': {gdf['damaged'].unique()}")
        print(gdf['damaged'].value_counts())
        gdf['damaged'].hist(bins=3)
        plt.title(f"Histograma de 'damaged' - {os.path.basename(gpkg_path)}")
        plt.xlabel('damaged')
        plt.ylabel('count')
        plt.show()
    else:
        print("No se encontró la columna 'damaged'.")
    print(f"Resumen de geometrías: {gdf.geom_type.value_counts()}")
    gdf.plot(edgecolor='k', alpha=0.5)
    plt.title(f"Geometrías - {os.path.basename(gpkg_path)}")
    plt.show()

def main():
    print(f'Analizando archivos en: {EXAMPLE_SUBMISSION_DIR}')
    for fname in os.listdir(EXAMPLE_SUBMISSION_DIR):
        if fname.endswith('.gpkg'):
            analyze_gpkg_file(os.path.join(EXAMPLE_SUBMISSION_DIR, fname))

if __name__ == "__main__":
    main() 