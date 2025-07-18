import json
import re
from pathlib import Path

DATA_ROOT = Path("data/international-charter-eo4ai-etq-challenge/Training")

def list_image_dates(city_dir):
    collection_path = city_dir / "collection.json"
    if not collection_path.exists():
        print(f"[WARN] No collection.json en {city_dir}")
        return
    with open(collection_path) as f:
        collection = json.load(f)
    desc = collection.get("description", "")
    m = re.search(r"Date of Charter Activation: (\d{4}-\d{2}-\d{2})", desc)
    if not m:
        print(f"[WARN] No se encontró fecha de evento en {city_dir}")
        return
    event_date = m.group(1) + "T00:00:00Z"
    print(f"\nCiudad: {city_dir.name}")
    print(f"Fecha del evento: {event_date}")
    print(f"{'Escena':60} {'Fecha':25} {'Pre/Post'}")
    print("-"*95)
    for link in collection.get("links", []):
        if link.get("rel") != "item":
            continue
        scene_json = city_dir / link["href"]
        if not scene_json.exists():
            continue
        with open(scene_json) as sf:
            scene = json.load(sf)
        scene_date = scene.get("properties", {}).get("datetime", None)
        if not scene_date:
            m2 = re.search(r"(\d{8})", scene_json.name)
            if m2:
                scene_date = m2.group(1)[:4] + "-" + m2.group(1)[4:6] + "-" + m2.group(1)[6:8] + "T00:00:00Z"
        if not scene_date:
            continue
        prepost = "pre" if scene_date < event_date else "post"
        print(f"{scene_json.name:60} {scene_date:25} {prepost}")

def main():
    for city_dir in DATA_ROOT.iterdir():
        if not city_dir.is_dir() or not city_dir.name.startswith("Call_"):
            continue
        list_image_dates(city_dir)

if __name__ == "__main__":
    main() 