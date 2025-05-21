from deepface import DeepFace
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, glob
import sys

# === Step 1: Load and sanitize label mapping from CSV ===
csv_path = "archive/Dataset.csv"
df = pd.read_csv(csv_path)
df['id'] = df['id'].str.strip().str.lower()
df['label'] = df['label'].str.strip()
label_lookup = dict(zip(df['id'], df['label']))

# === Step 2: Set paths ===
test_dir = "archive/Faces/Faces"
db_path = "archive/Original Images/Original Images"
test_images = glob.glob(f"{test_dir}/*.jpg")

# === Step 3: Evaluation Loop ===
y_true = []
y_pred = []

print(f"🚀 Starting evaluation on {len(test_images)} images with VGG-Face + RetinaFace...")

for i, img_path in enumerate(test_images):
    filename = os.path.basename(img_path).lower()
    true_label = label_lookup.get(filename, "Unknown")

    short_name = filename[:40] + "..." if len(filename) > 43 else filename
    progress_msg = f"\r🔁 [{i+1}/{len(test_images)}] Processing: {short_name:<45}"
    sys.stdout.write(progress_msg)
    sys.stdout.flush()


    try:
        result = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,
            silent=True
        )

        print(f"✅ DeepFace processed: {true_label}")


        if len(result[0]) > 0 and result[0].iloc[0]['distance'] <= 0.55:
            identity_path = result[0].iloc[0]['identity']
            predicted_label = os.path.basename(os.path.dirname(identity_path))
        else:
            predicted_label = "Unknown"

    except Exception as e:
        predicted_label = "Unknown"
        print(f"❌ Error processing {filename} with VGG-Face: {e}")


    print()  # moves cursor to new line after loop ends


    y_true.append(true_label)
    y_pred.append(predicted_label)

print("✅ Evaluation complete!")

# === Step 4: Classification Report ===
print("📋 Classification Report:")
all_labels = sorted(list(set(y_true + y_pred)))
print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

# === Step 5: Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred, labels=all_labels)
plt.figure(figsize=(18, 16))
sns.heatmap(cm, xticklabels=all_labels, yticklabels=all_labels, cmap="Blues", annot=True, fmt="d")
plt.title("Confusion Matrix – ArcFace + RetinaFace")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
