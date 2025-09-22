import tensorflow as tf
import numpy as np
from utils import load_data
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
DATA_DIR = "data"
OUT_DIR = "saved_models"
models_to_eval = {
    "U-Net": "unet.h5",
    "ResU-Net": "resunet.h5",
    "Attention U-Net": "attn_unet.h5"
}
(_, _), (_, _), (test_x, test_y) = load_data(DATA_DIR)
test_x = test_x / 255.0
results = []
for name, file in models_to_eval.items():
    model_path = os.path.join(OUT_DIR, file)
    if not os.path.exists(model_path):
        print(f"Skipping {name}, file not found: {model_path}")
        continue
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = (model.predict(test_x) > 0.5).astype("int")
    y_true = test_y.flatten()
    y_pred = preds.flatten()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n {name}")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    results.append([name, precision, recall, f1])
if results:
    labels = ["Precision", "Recall", "F1 Score"]
    plt.figure(figsize=(8,5))
    for i, (name, p, r, f1) in enumerate(results):
        scores = [p, r, f1]
        plt.bar([x + i*0.25 for x in range(len(scores))], scores, width=0.25, label=name)
    plt.xticks([x + 0.25 for x in range(len(labels))], labels)
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_comparison.png")
    plt.show()