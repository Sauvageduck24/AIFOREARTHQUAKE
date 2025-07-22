import sys
import geopandas as gpd
from pathlib import Path
import fiona


def main():
    if len(sys.argv) < 2:
        print("Uso: python view_gpkg_content.py <ruta_al_gpkg>")
        sys.exit(1)
    gpkg_path = Path(sys.argv[1])
    if not gpkg_path.exists():
        print(f"El archivo {gpkg_path} no existe.")
        sys.exit(1)
    try:
        # Listar capas disponibles en el archivo GPKG
        print("\n[INFO] Capas disponibles en el archivo:")
        print(fiona.listlayers(gpkg_path))

        # Leer la primera capa por defecto si no se especifica
        layer_name = fiona.listlayers(gpkg_path)[0]
        print(f"\n[INFO] Leyendo la capa: {layer_name}")
        gdf = gpd.read_file(gpkg_path, layer=layer_name)

        # Verificar si 'fid' está presente, si no, asignar el índice como 'fid'
        if 'fid' not in gdf.columns:
            gdf['fid'] = gdf.index
            print("\n[INFO] La columna 'fid' no estaba presente. Se ha asignado el índice como 'fid'.")

        # Verificar si 'id' está presente en el esquema detectado por Fiona
        with fiona.open(gpkg_path) as src:
            schema_columns = src.schema['properties'].keys()
            if 'id' in schema_columns and 'id' not in gdf.columns:
                gdf['id'] = gdf.index
                print("\n[INFO] La columna 'id' detectada por Fiona se ha asignado al índice como 'id'.")

        # Mostrar información detallada del archivo
        print(f"\n[INFO] Archivo: {gpkg_path}")
        print(f"Filas: {len(gdf)}")
        print(f"Columnas: {list(gdf.columns)}")
        print(f"CRS: {gdf.crs}")
        print(f"Primeras filas:")
        print(f"Índice: {gdf.index}")
        print(gdf.head(10))

        # Mostrar los primeros 10 elementos con Fiona
        print("\n[INFO] Primeros 10 elementos con Fiona:")
        with fiona.open(gpkg_path) as src:
            for i, feature in enumerate(src):
                print(feature)
                if i >= 20:
                    break

        # Mostrar valores únicos en la columna 'damaged' si está presente
        if 'damaged' in gdf.columns:
            print(f"\nValores únicos en la columna 'damaged': {gdf['damaged'].unique()}")
        else:
            print("\nLa columna 'damaged' no está presente en el archivo.")

    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()