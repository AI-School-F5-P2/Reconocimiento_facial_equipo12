import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0 

face_match = False
faces = []
similarity = 0

reference_img1 = cv2.imread("prueba1.jpg")
reference_img2 = cv2.imread("prueba2.jpg")

if reference_img1 is None or reference_img2 is None:
    print("Error al cargar imágenes de referencia.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error al cargar el clasificador Haarcascades.")

def check_face(frame):
    global face_match, faces, similarity
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        print("Número de caras detectadas:", len(faces))

        if len(faces) > 0:
            # Tomar la primera cara detectada para la verificación
            x, y, w, h = faces[0]
            face_roi = frame[y:y + h, x:x + w]

            similarity1 = DeepFace.verify(face_roi, reference_img1.copy())['verified']
            similarity2 = DeepFace.verify(face_roi, reference_img2.copy())['verified']

            # Si alguna de las imágenes coincide, considera que hay una coincidencia
            face_match = similarity1 or similarity2
            similarity = max(similarity1, similarity2)

    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al leer el frame de la cámara.")

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Dibujar el cuadro de enfoque alrededor de las caras (independientemente de la coincidencia de cara)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if face_match:
                similarity_text = f"Similarity: {int(similarity * 100)}%"
                cv2.putText(frame, similarity_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, "¡Welcome to the party!", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unidentified, please register", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("video", frame)
        cv2.waitKey(1)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()