import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import load_data
from models.unet import unet_model
from models.resnet import resunet_model
from models.attn_unet import attn_unet_model
from losses import bce_dice_loss
import pandas as pd
DATA_DIR = "data"   
OUT_DIR = "saved_models"
os.makedirs(OUT_DIR, exist_ok=True)
(train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(DATA_DIR)
train_x, val_x, test_x = train_x / 255.0, val_x / 255.0, test_x / 255.0
models = {
    "unet": unet_model,
    "resunet": resunet_model,
    "attn_unet": attn_unet_model,
}
results = []
for name, model_fn in models.items():
    print(f"\n=== TRAINING {name} ===")
    model = model_fn(input_shape=train_x.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss=bce_dice_loss,
                  metrics=["accuracy"])
    model_path = os.path.join(OUT_DIR, f"{name}.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=20,
        batch_size=8,
        callbacks=[checkpoint, earlystop]
    )
    best = tf.keras.models.load_model(model_path, compile=False)
    preds = (best.predict(test_x) > 0.5).astype("int")
    y_true = test_y.flatten()
    y_pred = preds.flatten()
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name} -> Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f}")
    results.append({"model": name, "precision": prec, "recall": rec, "f1": f1})
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT_DIR, "model_compare_results.csv"), index=False)
print("Saved comparison CSV to", os.path.join(OUT_DIR, "model_compare_results.csv"))