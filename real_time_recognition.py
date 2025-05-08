import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import glob
import tempfile
import threading
import time
from deepface import DeepFace

# Configuration
KNOWN_FACES_DIR = "dataset/known_faces"
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
THRESHOLD = 0.55
UNKNOWN_LOG_DIR = "unknown_logs"

# GUI appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Face Recognition")
        self.geometry("1280x720")
        self.resizable(False, False)

        self.model = None
        self.running = False
        self.cap = None
        self.last_unknown_log_time = 0  # prevent spam logging

        self.status_text = ctk.CTkLabel(self, text="Initializing...", font=("Arial", 18))
        self.status_text.pack(pady=10)

        blank_img = ImageTk.PhotoImage(Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)))
        self.video_panel = ctk.CTkLabel(self, image=blank_img, text="")
        self.video_panel.image = blank_img
        self.video_panel.pack()

        threading.Thread(target=self.initialize_and_start, daemon=True).start()

    def update_status(self, text):
        self.status_text.configure(text=text)
        self.status_text.update()

    def initialize_and_start(self):
        try:
            self.update_status("Loading DeepFace model...")
            self.model = DeepFace.build_model(MODEL_NAME)
            self.update_status("Model loaded.")

            image_candidates = glob.glob(f"{KNOWN_FACES_DIR}/**/*.jpg", recursive=True) + \
                               glob.glob(f"{KNOWN_FACES_DIR}/**/*.jpeg", recursive=True)

            if not image_candidates:
                self.update_status("No known faces found.")
                return

            self.update_status("Warming up embedding cache...")
            DeepFace.find(
                img_path=image_candidates[0],
                db_path=KNOWN_FACES_DIR,
                model_name=MODEL_NAME,
                enforce_detection=False,
                silent=True,
                detector_backend=DETECTOR
            )

            self.update_status("Ready. Starting camera...")
            time.sleep(1)
            self.running = True
            self.start_video_loop()

        except Exception as e:
            self.update_status(f"Error: {e}")

    def start_video_loop(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.update_status("Failed to open webcam.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            display_frame = frame.copy()

            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_image_path = tmp_file.name
                cv2.imwrite(temp_image_path, frame)

            try:
                faces = DeepFace.extract_faces(
                    img_path=temp_image_path,
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                    align=False
                )
            finally:
                os.remove(temp_image_path)

            for face_info in faces:
                area = face_info["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                face_crop = frame[y:y+h, x:x+w]

                try:
                    result = DeepFace.find(
                        img_path=face_crop,
                        db_path=KNOWN_FACES_DIR,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        silent=True,
                        detector_backend=DETECTOR
                    )

                    if len(result[0]) > 0:
                        top_result = result[0].iloc[0]
                        distance = top_result['distance']
                        if distance <= THRESHOLD:
                            identity_path = top_result['identity']
                            person_name = os.path.basename(os.path.dirname(identity_path))
                        else:
                            person_name = "Unknown"
                    else:
                        person_name = "Unknown"

                except:
                    person_name = "Unknown"
                    distance = None

                # Draw label on the frame
                color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                label = f"{person_name}" + (f" ({distance:.2f})" if distance is not None else "")
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Save unknown image with annotations
                if person_name == "Unknown":
                    now = time.time()
                    if now - self.last_unknown_log_time > 3:
                        self.last_unknown_log_time = now

                        os.makedirs(UNKNOWN_LOG_DIR, exist_ok=True)
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                        image_path = os.path.join(UNKNOWN_LOG_DIR, f"unknown_{timestamp}.jpg")
                        log_path = os.path.join(UNKNOWN_LOG_DIR, "unknown_log.txt")

                        # Save the current frame with bounding boxes
                        cv2.imwrite(image_path, display_frame)
                        with open(log_path, "a") as f:
                            f.write(f"{timestamp} - Unknown detected - saved to {image_path}\n")

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.video_panel.configure(image=img_tk)
            self.video_panel.image = img_tk
            self.update()

        self.cap.release()

    def on_close(self):
        self.running = False
        self.destroy()


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
