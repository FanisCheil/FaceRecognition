import customtkinter as ctk
import cv2
import os
import time
from PIL import Image, ImageTk
import pyttsx3
import threading

# Configuration
VIDEO_PATH = "face_prompt.mp4"
SAVE_DIR = "dataset/known_faces"
CAPTURE_INTERVAL = 0.4  # seconds

# Voice engine
engine = pyttsx3.init()
def speak(text):
    print("[Voice] " + text)
    engine.say(text)
    engine.runAndWait()

# GUI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceCaptureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Add Person to Dataset")
        self.geometry("1280x720")
        self.resizable(False, False)

        self.name = ""
        self.img_id = 1
        self.last_capture_time = time.time()
        self.capture_active = False

        # UI layout
        self.label = ctk.CTkLabel(self, text="Enter name of person:", font=("Arial", 20))
        self.label.pack(pady=10)

        self.entry = ctk.CTkEntry(self, width=300, placeholder_text="e.g. Fanis")
        self.entry.pack(pady=5)

        self.start_btn = ctk.CTkButton(self, text="Start Capture", command=self.start_capture)
        self.start_btn.pack(pady=15)

        self.video_frame = ctk.CTkLabel(self, text="")
        self.video_frame.pack(pady=10)

    def start_capture(self):
        self.name = self.entry.get().strip()
        if not self.name:
            ctk.CTkMessagebox(title="Error", message="Please enter a name.", icon="cancel")
            return

        self.person_dir = os.path.join(SAVE_DIR, self.name)
        os.makedirs(self.person_dir, exist_ok=True)
        self.start_btn.configure(state="disabled")
        speak("Follow the video prompts and look in all directions.")

        self.capture_active = True
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def capture_loop(self):
        video_cap = cv2.VideoCapture(VIDEO_PATH)
        cam_cap = cv2.VideoCapture(0)

        if not video_cap.isOpened() or not cam_cap.isOpened():
            print("❌ Could not open video or webcam.")
            return

        while True:
            ret_vid, video_frame = video_cap.read()
            ret_cam, cam_frame = cam_cap.read()

            if not ret_vid or not ret_cam:
                break

            # Resize both to fit side by side
            video_resized = cv2.resize(video_frame, (640, 480))
            cam_resized = cv2.resize(cam_frame, (640, 480))
            combined = cv2.hconcat([video_resized, cam_resized])
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(combined_rgb)
            img_tk = ImageTk.PhotoImage(img)

            # Show in label
            self.video_frame.configure(image=img_tk)
            self.video_frame.image = img_tk

            # Save frame if time passed
            if time.time() - self.last_capture_time >= CAPTURE_INTERVAL:
                img_path = os.path.join(self.person_dir, f"{self.name}_{self.img_id}.jpg")
                cv2.imwrite(img_path, cam_frame)
                print(f"📸 Saved: {img_path}")
                self.img_id += 1
                self.last_capture_time = time.time()

            # Delay slightly
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_cap.release()
        cam_cap.release()
        speak("Capture session complete.")
        print("🎉 Done")

# Launch
if __name__ == "__main__":
    app = FaceCaptureApp()
    app.mainloop()
