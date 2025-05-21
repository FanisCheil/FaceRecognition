import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Configuration
TEST_DIR = "dataset/test_faces"
KNOWN_FACES_DIR = "dataset/known_faces"
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
thresholds_to_test = [0.55]

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Get all class names from test set
class_names = sorted(os.listdir(TEST_DIR))
if "Unknown" not in class_names:
    class_names.append("Unknown")

for THRESHOLD in thresholds_to_test:
    print(f"\n📐 Testing with THRESHOLD = {THRESHOLD}\n{'-'*40}")

    y_true = []
    y_pred = []
    detailed_results = []

    image_paths = glob.glob(f"{TEST_DIR}/**/*.jpg", recursive=True) + \
                  glob.glob(f"{TEST_DIR}/**/*.jpeg", recursive=True)

    for img_path in image_paths:
        actual_label = os.path.basename(os.path.dirname(img_path))
        y_true.append(actual_label)

        predicted_label = "Unknown"
        match_distance = None

        try:
            result = DeepFace.find(
                img_path=img_path,
                db_path=KNOWN_FACES_DIR,
                model_name=MODEL_NAME,
                enforce_detection=False,
                silent=True,
                detector_backend=DETECTOR
            )

            if len(result[0]) > 0:
                top = result[0].iloc[0]
                predicted_label = os.path.basename(os.path.dirname(top['identity']))
                match_distance = top['distance']

                if match_distance <= THRESHOLD:
                    y_pred.append(predicted_label)
                else:
                    predicted_label = "Unknown"
                    y_pred.append("Unknown")
            else:
                y_pred.append("Unknown")

        except Exception as e:
            print(f"⚠️ Error with {img_path}: {e}")
            y_pred.append("Unknown")

        # Log to detailed results
        detailed_results.append({
            "filename": os.path.basename(img_path),
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "distance": match_distance,
            "match": actual_label == predicted_label
        })

    # 📊 Print classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, labels=class_names, zero_division=0))

    # 🧱 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 📉 Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix @ Threshold = {THRESHOLD}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 📋 Save and preview detailed results
    results_df = pd.DataFrame(detailed_results)
    print("\n📋 Sample of Detailed Results:")
    print(results_df.head(10))  # Show top 10 results

    csv_path = f"output/deepface_results_threshold_{THRESHOLD}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to: {csv_path}")
