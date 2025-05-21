import os
from deepface import DeepFace
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Load VGG-Face model and unwrap
vggface_wrapper = DeepFace.build_model("VGG-Face")
model = vggface_wrapper.model

# Load and preprocess image
img_path = "dataset/known_faces/Fanis/Fanis_5.jpg"  # Update this if needed
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = np.expand_dims(img_rgb, axis=0).astype("float32")

# Get list of all conv layers
conv_layer_names = [layer.name for layer in model.layers if "conv2d" in layer.name]

# Create output directory
output_dir = "vggface_gradcam_layers"
os.makedirs(output_dir, exist_ok=True)

# Run Grad-CAM for each conv layer
for layer_name in conv_layer_names:
    try:
        last_conv_layer = model.get_layer(layer_name)
        heatmap_model = Model([model.input], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = heatmap_model(img_input)
            loss = tf.reduce_mean(predictions)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-6)

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

        # Save the image
        cv2.imwrite(os.path.join(output_dir, f"{layer_name}.png"), superimposed)

        print(f"✅ Saved: {layer_name}.png")

    except Exception as e:
        print(f"❌ Failed on layer {layer_name}: {e}")
