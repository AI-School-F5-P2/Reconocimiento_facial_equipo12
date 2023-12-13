import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0 

face_match = False

reference_img = cv2.imread("prueba1.jpg")
if reference_img is None:
    print("Error al cargar imagen.")
    exit()


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
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

        if face_match:
            cv2.putText(frame, "¡Welcome to the party!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haardcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x + w,y + h), (0,255,0),2)

        else:
            cv2.putText(frame, "Unidentified, please register", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        
        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()