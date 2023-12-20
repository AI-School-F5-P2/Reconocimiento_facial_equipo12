import face_recognition
import os
import cv2
import numpy as np
import streamlit as st

# Porcentaje de precisión
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%" 
    else:
        value = (linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"

# Clase reconocimiento facial
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir("faces"):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def run_recognition(self, frame):
        # Cambiar el tamaño y pasar el frame a RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Encontrar todas las caras en el frame
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            self.face_names.append(f"{name} ({confidence})")

        self.process_current_frame = not self.process_current_frame

        # Dibujar el recuadro y mostrar el nombre arriba
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Verificar si el nombre es conocido
            if "Unknown" not in name:
                # Acceso permitido, recuadro verde
                rectangle_color = (0, 255, 0)
                # Extraer el nombre sin la extensión .png
                name = name.split('.')[0]
            else:
                # Acceso no permitido, recuadro rojo
                rectangle_color = (0, 0, 255)

            # Dibujar el recuadro 
            cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
            # Mostrar el nombre arriba
            cv2.putText(frame, f"{name} ({confidence})", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Agregar el texto "Acceso Permitido" si el nombre es conocido
            if "Unknown" not in name:
                cv2.putText(frame, "Acceso Permitido", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

        return frame

def main():
    fr = FaceRecognition()

    st.title("FIESTA FACTORIA F5")
    st.markdown('<style>h1 { color: #FFA500; }</style>', unsafe_allow_html=True)
    # Agregar una imagen al lado del título
    st.image("https://e00-expansion.uecdn.es/assets/multimedia/imagenes/2018/12/14/15448133643420.jpg", caption="fiesta navidad 2023", use_column_width=True)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("¡Error: No se puede abrir la fuente de video!")

    st.header("Identifícate, por favor!!")
    video_placeholder = st.empty()

    while True:
        ret, frame = video_capture.read()
        if ret:
            frame = fr.run_recognition(frame)

            # Mostrar el video en Streamlit
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Cerrar la aplicación cuando se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
