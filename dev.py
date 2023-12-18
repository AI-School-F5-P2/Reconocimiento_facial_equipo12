import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import numpy as np
from deepface import DeepFace

class FaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fiesta de fin de año Factoria 2023")

        self.label = tk.Label(root, text="Fiesta de fin de año Factoria 2023")
        self.label.pack(pady=10)

        self.button = tk.Button(root, text="Acceder", command=self.start_face_detection)
        self.button.pack(pady=10)

        self.cap = None
        self.face_match = False
        self.similarity = 0

        # Inicialización del modelo de detección facial integrado
        self.net = cv2.dnn.readNet(cv2.samples.findFile('res10_300x300_ssd_iter_140000.caffemodel'), 
                                   cv2.samples.findFile('deploy.prototxt'))

        # Cargar las imágenes de referencia
        self.reference_img1 = cv2.imread("prueba1.jpg")
        self.reference_img2 = cv2.imread("prueba2.jpg")

        if self.reference_img1 is None or self.reference_img2 is None:
            messagebox.showerror("Error", "Error al cargar imágenes de referencia.")
            root.destroy()
            return

    def start_face_detection(self):
        self.label.pack_forget()
        self.button.pack_forget()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

        threading.Thread(target=self.video_stream, daemon=True).start()

    def check_face(self, frame):
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (x, y, w, h) = box.astype("int")
                    face_roi = frame[y:y + h, x:x + w]

                    similarity1 = DeepFace.verify(face_roi, self.reference_img1.copy())['verified']
                    similarity2 = DeepFace.verify(face_roi, self.reference_img2.copy())['verified']

                    self.face_match = similarity1 or similarity2
                    self.similarity = max(similarity1, similarity2)

                    self.draw_bounding_box(frame, (x, y, w, h))
                    self.draw_result_text(frame, (x, y, w, h))

        except ValueError:
            self.face_match = False

    def video_stream(self):
        _, frame = self.cap.read()

        if frame is not None:
            self.check_face(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)

            if not hasattr(self, 'panel'):
                self.panel = tk.Label(self.root, image=img)
                self.panel.image = img
                self.panel.pack(side="top", padx=10, pady=10)
            else:
                self.panel.configure(image=img)
                self.panel.image = img

            self.root.after(10, self.video_stream)  # Reemplaza cv2.waitKey(1)

    def draw_bounding_box(self, frame, face_coordinates):
        x, y, w, h = face_coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_result_text(self, frame, face_coordinates):
        x, y, w, h = face_coordinates
        if self.face_match:
            result_text = "¡Bienvenido a la fiesta!"
            similarity_text = f"Similarity: {int(self.similarity * 100)}%"
            cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, similarity_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            result_text = "No identificado, por favor regístrese"
            cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectorApp(root)
    root.mainloop()