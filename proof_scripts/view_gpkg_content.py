import sys
import geopandas as gpd
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Uso: python view_gpkg_content.py <ruta_al_gpkg>")
        sys.exit(1)
    gpkg_path = Path(sys.argv[1])
    if not gpkg_path.exists():
        print(f"El archivo {gpkg_path} no existe.")
        sys.exit(1)
    try:
        gdf = gpd.read_file(gpkg_path)
        print(f"\n[INFO] Archivo: {gpkg_path}")
        print(f"Filas: {len(gdf)}")
        print(f"Columnas: {list(gdf.columns)}")
        print(f"CRS: {gdf.crs}")
        print(f"Primeras filas:")
        print(gdf.head())
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 