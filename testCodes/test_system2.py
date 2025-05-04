import cv2
import os
import glob
import tempfile
from deepface import DeepFace

# Configuration
KNOWN_FACES_DIR = "dataset/known_faces"
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
THRESHOLD = 0.55

# Load model (trigger GPU warm-up)
print("🧠 Loading DeepFace model...")
DeepFace.build_model(MODEL_NAME)

# Warm-up known face embeddings
image_candidates = glob.glob(f"{KNOWN_FACES_DIR}/**/*.jpg", recursive=True) + \
                   glob.glob(f"{KNOWN_FACES_DIR}/**/*.jpeg", recursive=True)

if not image_candidates:
    print("❌ No face images found in known_faces!")
    exit()

first_image = image_candidates[0]

print("🔄 Warming up embedding cache...")
DeepFace.find(
    img_path=first_image,
    db_path=KNOWN_FACES_DIR,
    model_name=MODEL_NAME,
    enforce_detection=False,
    silent=True,
    detector_backend=DETECTOR
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to connect to camera stream.")
    exit()

print("📸 Camera stream connected")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No frame received")
        break

    # Save current frame temporarily (required by extract_faces)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        cv2.imwrite(temp_image_path, frame)

    # Detect faces using RetinaFace
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

        # Crop the face directly from the original frame
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

        except Exception as e:
            person_name = "Unknown"
            distance = None

        # Choose box color
        if person_name == "Unknown":
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{person_name}" + (f" ({distance:.2f})" if distance is not None else "")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
