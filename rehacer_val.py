from pathlib import Path
import random
import shutil

random.seed(0)

salida = Path("dataset_yolo")
carpeta_train = salida / "train"
carpeta_val = salida / "val"

for foto in list(carpeta_val.rglob("*.jpg")):  #regresar las fotos almacenadas
    destino = carpeta_train / foto.parent.name
    destino.mkdir(exist_ok=True)
    nombre = foto.name
    if (destino / nombre).exists():
        nombre = "val_" + nombre
    shutil.move(foto, destino / nombre)

shutil.rmtree(carpeta_val)

for carpeta in sorted(carpeta_train.iterdir()):
    if not carpeta.is_dir():
        continue
    fotos = sorted(carpeta.glob("*.jpg"))
    random.shuffle(fotos)
    cantidad = max(1, round(len(fotos) * 0.15))

    destino = carpeta_val / carpeta.name
    destino.mkdir(parents=True, exist_ok=True)
    for foto in fotos[:cantidad]:
        shutil.move(foto, destino / foto.name)

# para rehacer los caches
for cache in salida.glob("*.cache"):
    cache.unlink()

for particion in ["train", "val", "test"]:
    base = salida / particion #verificar la salida de los datos 
    clases = [c for c in base.iterdir() if c.is_dir()]
    print(particion, "->", len(clases), "clases,", len(list(base.rglob("*.jpg"))), "fotos")