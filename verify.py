import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from deepface import DeepFace
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load LFW test set (1000 pairs: 500 same, 500 different)
print("📦 Downloading LFW test pairs (1,000 pairs: 500 same, 500 different)...")
lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0, funneled=True)
images = lfw.pairs
labels = lfw.target  # 1 = same person, 0 = different

print("🚀 Starting verification with DeepFace...\n")

predictions = []
distances = []
probas = []

start_time = time.time()

for i, (img1, img2) in enumerate(images):
    img1 = (img1 * 255).astype("uint8")
    img2 = (img2 * 255).astype("uint8")

    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="VGG-Face",
            detector_backend="mtcnn",
            enforce_detection=True,
            silent=True
        )
        predicted = int(result["verified"])
        distance = result["distance"]
        score = 1 - distance  # For ROC: similarity score

    except Exception as e:
        predicted = 0
        distance = None
        score = 0  # Treat as very dissimilar

    predictions.append(predicted)
    distances.append(distance)
    probas.append(score)

    print(f"[{i+1:>4}/{len(images)}] ✅ GT: {labels[i]} | Pred: {predicted} | Dist: {distance}")

end_time = time.time()

# Accuracy
accuracy = accuracy_score(labels, predictions)
print(f"\n✅ DeepFace (VGG-Face) manual verification accuracy on LFW: {accuracy * 100:.2f}%")
print(f"🕒 Total time: {int(end_time - start_time)} seconds")

# Confusion Matrix
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Different", "Same"], yticklabels=["Different", "Same"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(labels, probas)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
