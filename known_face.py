import cv2
import os
import time
import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Configuration
SAVE_DIR = "dataset/known_faces"
POSES = ["Forward", "Left", "Right", "Up", "Down","Forward and raise eyebrows", "and Smile"]
CAPTURES_PER_POSE = 3

# Prompt for name
name = input("Enter your name: ").strip()
person_dir = os.path.join(SAVE_DIR, name)
os.makedirs(person_dir, exist_ok=True)

print("📸 Starting face capture session...")
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

img_id = 1

for pose in POSES:
    for i in range(CAPTURES_PER_POSE):
        speak(f"Look {pose}")
        print(f"\n🧭 Pose: Look {pose} ({i + 1}/{CAPTURES_PER_POSE})")
        speak("Hold still. Capturing in 2 seconds...")

        time.sleep(2)

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture image.")
            continue

        img_path = os.path.join(person_dir, f"{name}_{img_id}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved: {img_path}")
        img_id += 1

speak("Capture session complete.")
print("🎉 Capture session complete.")
cap.release()
