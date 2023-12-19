import tensorflow as tf
import cv2
import streamlit as st
import base64
import threading
from deepface import DeepFace


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background-image: url("https://img.freepik.com/fotos-premium/celebracion-fin-ano-champagne_93675-50557.jpg");
  background-size: 100%;
  background-position: top left;
  background-repeat: no-repeat;
  background-attachment: local;
  filter: brightness(1.2); /* Aumente el brillo en un 20% */
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("BIENVENID@!!!")

with st.container():
    st.header("Identificate, por favor")

# Inicializar la cámara web
try:
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Intenta con CAP_MSMF
    if not camera.isOpened():
        raise cv2.error("No se puede abrir la cámara.")
except cv2.error as e:
    print(f"Error al abrir la cámara: {e}")
    st.error("Error al abrir la cámara.")
    st.stop()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables para la verificación facial
face_match = False
similarity = 0
faces = []
lock = threading.Lock()

# Cargar imágenes de referencia
reference_img1 = cv2.imread("prueba1.jpg")
reference_img2 = cv2.imread("prueba2.jpg")

if reference_img1 is None or reference_img2 is None:
    st.error("Error al cargar imágenes de referencia.")
    st.stop()

# Cargar el clasificador Haarcascades para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    st.error("Error al cargar el clasificador Haarcascades.")
    st.stop()

# Crear una ventana para mostrar el video en Streamlit
frame_placeholder = st.empty()

# Función para verificar la cara en un hilo separado
def check_face(frame):
    global face_match, faces, similarity
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        local_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        with lock:
            faces = local_faces.copy()

        if len(local_faces) > 0:
            # Tomar la primera cara detectada para la verificación
            x, y, w, h = local_faces[0]
            face_roi = frame[y:y + h, x:x + w]

            # Convertir el recorte del rostro a RGB para DeepFace
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Realizar la verificación facial
            result = DeepFace.verify(face_rgb, reference_img1.copy(), model_name="Facenet", enforce_detection=False) or \
                     DeepFace.verify(face_rgb, reference_img2.copy(), model_name="Facenet", enforce_detection=False)

            # Si alguna de las imágenes coincide, considera que hay una coincidencia
            with lock:
                face_match = result['verified']
                similarity = result['similarity']

    except Exception as e:
        print("Error en la verificación facial:", str(e))
        face_match = False

# Iniciar la aplicación web
run = st.checkbox('Run')

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Verificar la cara en un hilo separado
    threading.Thread(target=check_face, args=(frame.copy(),)).start()

    # Dibujar el cuadro de enfoque alrededor de las caras (independientemente de la coincidencia de cara)
    with lock:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if face_match:
                similarity_text = f"Similarity: {int(similarity * 100)}%"
                cv2.putText(frame, similarity_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "¡Welcome to the party!", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unidentified, please register", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Mostrar el fotograma en la interfaz de la aplicación web
    frame_placeholder.image(frame)

# Detener la cámara y cerrar la aplicación cuando se desactive la casilla de verificación
camera.release()
st.write('Stopped')
