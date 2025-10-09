import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import f1_score
#  Utility: Safe Load Model
def safe_load_model(path):
    try:
        print(f" Loading model from {path}")
        return tf.keras.models.load_model(path, compile=False)
    except TypeError as e:
        if "Unknown dtype policy" in str(e):
            print(" TensorFlow 2.19 deserialization issue detected. Applying custom dtype patch...")
            from keras import mixed_precision
            custom_objects = {"DTypePolicy": mixed_precision.Policy("float32")}
            with custom_object_scope(custom_objects):
                return tf.keras.models.load_model(path, compile=False)
        else:
            raise e
    except Exception as e:
        print(f" Model load failed: {e}")
        raise e
#  Test Data Loading
print("📥 Loading test dataset...")
x_test_path = os.path.join("data", "test", "testX.npy")
y_test_path = os.path.join("data", "test", "testY.npy")
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
#  Model Evaluation Function
def evaluate_model(model, x_test, y_test, model_name="Model"):
    preds = model.predict(x_test, verbose=1)
    preds_binary = (preds > 0.5).astype(np.uint8)
    acc = np.mean(preds_binary == y_test)
    f1 = f1_score(y_test.flatten(), preds_binary.flatten())
    print(f"📊 {model_name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return acc, f1
#  Evaluate Models & Write Results
results = []
# List of models to evaluate
model_files = {
    "U-Net": "unet.h5",
    "ResU-Net": "resunet_model.h5",
    "Attention U-Net": "attnunet_model.h5",
    "Attention Res U-Net": "attn_resunet.h5",
    "ASDMS U-Net": "adsms_unet.h5"
}
for model_name, model_file in model_files.items():
    try:
        print(f"\n Evaluating {model_name} ...")
        model_path = os.path.join("saved_models", model_file)
        model = safe_load_model(model_path)
        acc, f1 = evaluate_model(model, x_test, y_test, model_name)
        results.append(f"{model_name} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    except Exception as e:
        error_msg = f" Could not evaluate {model_name}: {e}"
        print(error_msg)
        results.append(error_msg)
#  Write Results to File
results_path = "results.txt"
with open(results_path, "w") as f:
    for line in results:
        f.write(line + "\n")
print(f"Results written to {results_path}")
