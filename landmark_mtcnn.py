from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt

# Image paths
image_paths = [
    "dataset/known_faces/Fanis/Fanis_5.jpg",
    "dataset/known_faces/Fanis/Fanis_136.jpg",
    "dataset/known_faces/Fanis/Fanis_152.jpg"
]

# MTCNN detector
detector = MTCNN()

# Setup figure
plt.figure(figsize=(15, 12))

for row_idx, img_path in enumerate(image_paths):
    # Load and convert image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_bbox = img_rgb.copy()
    img_landmarks = img_rgb.copy()

    # Detect faces
    detections = detector.detect_faces(img_rgb)

    for detection in detections:
        x, y, w, h = detection['box']
        keypoints = detection['keypoints']

        # Draw bounding box
        cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_landmarks, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks
        for point in keypoints.values():
            cv2.circle(img_landmarks, point, 4, (255, 0, 0), -1)

    # Plot all 3 versions
    plt.subplot(3, 3, row_idx * 3 + 1)
    plt.imshow(img_rgb)
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
