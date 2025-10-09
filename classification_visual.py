import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
# Paths
DATA_DIR = os.getcwd()
MODEL_DIR = os.path.join(DATA_DIR, "saved_models")
x_test_path = os.path.join("data", "test", "testX.npy")
y_test_path = os.path.join("data", "test", "testY.npy")
# Load Test Data
print("📥 Loading test data...")
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)
print(f"✅ Loaded test data: x_test={x_test.shape}, y_test={y_test.shape}\n")
# Load All Models
print(" Loading models...")
models_dict = {
    "U-Net": load_model(os.path.join(MODEL_DIR, "unet.h5"), compile=False),
    "ResU-Net": load_model(os.path.join(MODEL_DIR, "resunet_model.h5"), compile=False),
    "Attention U-Net": load_model(os.path.join(MODEL_DIR, "attnunet_model.h5"), compile=False),
    "Attention ResU-Net": load_model(os.path.join(MODEL_DIR, "attn_resunet.h5"), compile=False),
    "ADSMS U-Net": load_model(os.path.join(MODEL_DIR, "adsms_unet.h5"), compile=False),
}
# Predict for all models
print(" Running predictions...")
predictions = {}
for name, model in models_dict.items():
    print(f"→ Predicting with {name} ...")
    y_pred = (model.predict(x_test, batch_size=8, verbose=1) > 0.5).astype(np.uint8)
    predictions[name] = y_pred
# Compute Confusion Matrices for all models
print("\n Generating confusion matrices...")
y_true_flat = y_test.flatten()
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
colors = ['Blues', 'Greens', 'Purples', 'Oranges', 'Reds']
for idx, (name, y_pred) in enumerate(predictions.items()):
    y_pred_flat = y_pred.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    sns.heatmap(cm, annot=True, fmt='d', cmap=colors[idx], ax=axes[idx])
    axes[idx].set_title(f"{name}")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
plt.tight_layout()
plt.show()
# Bar Chart for Metrics (from your results)
print("\n📈 Creating metrics comparison chart...")
data = {
    "Model": ["U-Net", "ResU-Net", "Attention U-Net", "Attention ResU-Net", "ADSMS U-Net"],
    "Accuracy": [0.9479, 0.9359, 0.9397, 0.9375, 0.9401],
    "F1": [0.7237, 0.6739, 0.7357, 0.6573, 0.6743],
}
df = pd.DataFrame(data)
metrics = ["Accuracy", "F1"]
x = np.arange(len(metrics))
width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))
# Plot each model
for i in range(len(df)):
    ax.bar(x + (i - 2) * width, df.loc[i, metrics], width, label=df.loc[i, "Model"])
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison (Accuracy & F1)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.show()
print("\n Evaluation & visualization complete!")
