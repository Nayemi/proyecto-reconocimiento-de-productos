import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Reconocimiento de productos", page_icon="c")
st.title("Reconocimiento de productos de supermercado")

@st.cache_resource
def cargar_modelo():
    return YOLO("modelos/best.pt") #d

modelo = cargar_modelo()
archivo = st.file_uploader("Sube una foto del producto", type=["jpg", "jpeg", "png"]) #archivos que si acepta

if archivo:
    # FOTO
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Tu foto", width=350)

    # hacer las comparaciones, predicciones, etc
    resultado = modelo(imagen, verbose=False)[0]
    nombre = resultado.names[resultado.probs.top1]
    confianza = float(resultado.probs.top1conf)

    st.subheader(nombre.replace("-", " "))
    st.write(f"Confianza: {confianza:.1%}")

    if confianza < 0.5:
        st.warning("El modelo no está muy seguro. Prueba con otra foto.")

    #probabilidad o compajinidad que tiene con los productos del data set
    indices = resultado.probs.top5[:3]
    confs = resultado.probs.top5conf[:3].tolist()
    tabla = pd.DataFrame({
        "Producto": [resultado.names[i] for i in indices],
        "Probabilidad": confs,
    }).set_index("Producto")
    st.bar_chart(tabla)