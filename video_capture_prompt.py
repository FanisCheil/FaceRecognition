import customtkinter as ctk
import cv2
import os
import time
from PIL import Image, ImageTk
import pyttsx3
import threading
import argparse
import shutil
import tkinter.messagebox  

# Configuration
VIDEO_PATH = "face_prompt.mp4"
SAVE_DIR = "dataset/known_faces"
CAPTURE_INTERVAL = 0.7  # seconds

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
    def __init__(self, preset_name=None, update_mode=False):
        super().__init__()
        self.title("Add Person to Dataset" if not update_mode else "Update Person")
        self.geometry("1280x720")
        self.resizable(False, False)

        self.name = preset_name if preset_name else ""
        self.update_mode = update_mode
        self.img_id = 1
        self.last_capture_time = time.time()
        self.capture_active = False

        # UI layout
        self.label = ctk.CTkLabel(self, text="Enter name of person:", font=("Arial", 20))
        self.label.pack(pady=10)

        self.entry = ctk.CTkEntry(self, width=300, placeholder_text="e.g. YourName")
        self.entry.pack(pady=5)
        if self.name:
            self.entry.insert(0, self.name)
            self.entry.configure(state="disabled")

        self.start_btn = ctk.CTkButton(self, text="Start Capture", command=self.start_capture)
        self.start_btn.pack(pady=15)

        self.video_frame = ctk.CTkLabel(self, text="")
        self.video_frame.pack(pady=10)

        # Auto-start if update mode
        if self.name and self.update_mode:
            self.start_capture()

    def start_capture(self):
        if self.update_mode:

            self.person_dir = os.path.join(SAVE_DIR, self.name)
            # Use preset name (already set)
            if os.path.exists(self.person_dir):
                shutil.rmtree(self.person_dir)
            os.makedirs(self.person_dir)
        else:
            # Always read name fresh from entry
            self.name = self.entry.get().strip()
            if not self.name:
                tkinter.messagebox.showerror("Error", "Please enter a name.")
                return

            self.person_dir = os.path.join(SAVE_DIR, self.name)

            if os.path.exists(self.person_dir):
                tkinter.messagebox.showerror("Error", "This person already exists.")
                return
            os.makedirs(self.person_dir)

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

        self.img_id = 1
        self.last_capture_time = time.time()

        while True:
            ret_vid, video_frame = video_cap.read()
            ret_cam, cam_frame = cam_cap.read()

            if not ret_vid or not ret_cam:
                break

            video_resized = cv2.resize(video_frame, (640, 480))
            cam_resized = cv2.resize(cam_frame, (640, 480))
            combined = cv2.hconcat([video_resized, cam_resized])
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(combined_rgb)
            img_tk = ImageTk.PhotoImage(img)

            self.video_frame.configure(image=img_tk)
            self.video_frame.image = img_tk

            if time.time() - self.last_capture_time >= CAPTURE_INTERVAL:
                img_path = os.path.join(self.person_dir, f"{self.name}_{self.img_id}.jpg")
                cv2.imwrite(img_path, cam_frame)
                print(f"[Saved] {img_path}")
                self.img_id += 1
                self.last_capture_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_cap.release()
        cam_cap.release()
        speak("Capture session complete.")
        print("[Done] Capture finished.")
        self.start_btn.configure(state="normal")
        time.sleep(2) #wait 2 seconds
        self.destroy()

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Person name to update and overwrite")
    args = parser.parse_args()

    update_mode = args.name is not None

    app = FaceCaptureApp(preset_name=args.name, update_mode=update_mode)
    app.mainloop()
