from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# List of 3 image paths
image_paths = [
    "dataset/known_faces/Fanis/Fanis_5.jpg",
    "dataset/known_faces/Fanis/Fanis_136.jpg",
    "dataset/known_faces/Fanis/Fanis_152.jpg"
]

# Set up plot
plt.figure(figsize=(15, 12))

for row_idx, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)

    # 1. Original Image
    img_original = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # 2. Bounding box only
    img_bbox = img.copy()
    # 3. Bounding box + landmarks
    img_landmarks = img.copy()

    # Detection
    detections = RetinaFace.detect_faces(img_path)

    for key in detections:
        face = detections[key]
        x1, y1, x2, y2 = face["facial_area"]
        landmarks = face["landmarks"]

        # Bounding box
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_landmarks, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Landmarks
        for point in landmarks.values():
            cv2.circle(img_landmarks, tuple(map(int, point)), 4, (255, 0, 0), -1)

    # Convert to RGB
    img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
    img_landmarks = cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2RGB)

    # Plot all 3 versions for each image
    plt.subplot(3, 3, row_idx * 3 + 1)
    plt.imshow(img_original)
    plt.title(f"Image {row_idx + 1} - Original")
    plt.axis("off")

    plt.subplot(3, 3, row_idx * 3 + 2)
    plt.imshow(img_bbox)
    plt.title(f"Image {row_idx + 1} - Bounding Box")
    plt.axis("off")

    plt.subplot(3, 3, row_idx * 3 + 3)
    plt.imshow(img_landmarks)
    plt.title(f"Image {row_idx + 1} - Landmarks")
    plt.axis("off")

plt.tight_layout()
plt.show()
