from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    modelo = YOLO("modelos/best.pt")
    carpeta_test = Path("dataset_yolo/test")

    reales = []      # datos reales 
    predichas = []   # las predicciones

    for carpeta in sorted(carpeta_test.iterdir()):
        if not carpeta.is_dir():
            continue
        for foto in carpeta.glob("*.jpg"):
            resultado = modelo(str(foto), verbose=False)[0]
            reales.append(carpeta.name)
            predichas.append(resultado.names[resultado.probs.top1])


    acc = accuracy_score(reales, predichas)
    print(f"Accuracy en test: {acc:.3f}")

    reporte = classification_report(reales, predichas, zero_division=0)
    print(reporte)
    tabla = pd.DataFrame(
        classification_report(reales, predichas, output_dict=True, zero_division=0)
    ).T
    tabla.to_csv("reporte_clases.csv")

    etiquetas = sorted(set(reales))
    cm = confusion_matrix(reales, predichas, labels=etiquetas)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Cantidad de fotos")
    plt.title("Matriz de confusión (test)")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    plt.show()

    
    errores = cm.copy()
    np.fill_diagonal(errores, 0)    #quita los correctos
    top = np.argsort(errores, axis=None)[::-1][:5]
    print("\nConfusiones más frecuentes:")
    for idx in top:
        i, j = np.unravel_index(idx, errores.shape)
        if errores[i, j] > 0:
            print(f"  {etiquetas[i]} -> confundida con {etiquetas[j]}: {errores[i, j]} veces")