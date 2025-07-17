import json
from pathlib import Path
from collections import Counter

DATA_ROOT = Path("data/international-charter-eo4ai-etq-challenge/Training")
band_counter = Counter()

for city_dir in DATA_ROOT.iterdir():
    if not city_dir.is_dir() or not city_dir.name.startswith("Call_"):
        continue
    collection_path = city_dir / "collection.json"
    if not collection_path.exists():
        continue
    with open(collection_path) as f:
        collection = json.load(f)
    for link in collection.get("links", []):
        if link.get("rel") != "item":
            continue
        scene_json = city_dir / link["href"]
        if not scene_json.exists():
            continue
        with open(scene_json) as sf:
            scene = json.load(sf)
        assets = scene.get("assets", {})
        print(f"\nCiudad: {city_dir.name} | Escena: {scene_json.name}")
        print(f"Bandas encontradas: {list(assets.keys())}")
        band_counter.update(assets.keys())

print("\nResumen de nombres de bandas encontrados en todo el dataset:")
for band, count in band_counter.most_common():
    print(f"{band}: {count}") 