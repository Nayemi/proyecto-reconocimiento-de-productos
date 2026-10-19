from pathlib import Path
import shutil
import pandas as pd


data_base = Path("dataset/GroceryStoreDataset/dataset")
salida = Path("dataset_yolo")

def leer(nombre):
    filas = []
    for linea in (data_base / nombre).read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        ruta, fino, grueso = [p.strip() for p in linea.split(",")]
        filas.append({"ruta": ruta, "clase": Path(ruta).parent.name, "fino": int(fino)})
    return pd.DataFrame(filas)

for particion, archivo in [("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")]:
    df = leer(archivo)

    assert df.groupby("fino")["clase"].nunique().max() == 1, "Hay nombres repetidos"

    for _, fila in df.iterrows():
        destino = salida / particion / fila["clase"]
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copy(data_base / fila["ruta"], destino / Path(fila["ruta"]).name)

    print(particion, "listo:", len(df), "imágenes")

print("Carpeta creada en:", salida.resolve())