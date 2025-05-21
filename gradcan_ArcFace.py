from deepface import DeepFace
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load ArcFace model and unwrap Keras model
arcface_client = DeepFace.build_model("ArcFace")
arcface_model = arcface_client.model  # Unwrap internal model

# Load and preprocess image
img_path = "dataset/known_faces/Fanis/Fanis_5.jpg"  # <-- Update path
img = cv2.imread(img_path)
img = cv2.resize(img, (112, 112))  # ArcFace expects 112x112
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = np.expand_dims(img_rgb, axis=0).astype("float32")

# Create output folder
os.makedirs("gradcam_outputs", exist_ok=True)

# Filter all convolutional layers
conv_layers = [layer.name for layer in arcface_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

for layer_name in conv_layers:
    try:
        # Define model for Grad-CAM
        heatmap_model = Model(inputs=arcface_model.input,
                              outputs=[arcface_model.get_layer(layer_name).output, arcface_model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = heatmap_model(img_input)
            loss = tf.reduce_mean(predictions)

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        # Generate heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-6)
        heatmap = cv2.resize(heatmap, (112, 112))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

        # Save image
        out_path = os.path.join("arcface_gradcam_outputs", f"{layer_name}.png")
        cv2.imwrite(out_path, overlay)

        print(f"✅ Saved Grad-CAM for layer: {layer_name}")

    except Exception as e:
        print(f"❌ Error on layer {layer_name}: {e}")
