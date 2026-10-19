from pathlib import Path

for particion in ["train", "val", "test"]:
    base = Path("dataset_yolo") / particion
    carpetas = [c for c in base.iterdir() if c.is_dir()]
    fotos = list(base.rglob("*.jpg"))
    print(particion, "->", len(carpetas), "clases,", len(fotos), "fotos")