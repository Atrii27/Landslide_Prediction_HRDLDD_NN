import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm  
# PATHS
MODEL_PATHS = {
    "U-Net": "saved_models/unet.h5",
    "ResU-Net": "saved_models/resunet_model.h5",
    "Attn U-Net": "saved_models/attnunet_model.h5",
    "Attn ResU-Net": "saved_models/attn_resunet.h5",
    "ADSMS U-Net": "saved_models/adsms_unet.h5"
}
TEST_DATA_DIR = "data/test/"
SAVE_DIR = "results_imgs/"
# Create folder for results
os.makedirs(SAVE_DIR, exist_ok=True)
# LOAD TEST DATA
print("Loading test data...")
X_test = np.load(os.path.join(TEST_DATA_DIR, "testX.npy"))  # satellite images
y_test = np.load(os.path.join(TEST_DATA_DIR, "testY.npy"))  # binary masks
print(f" Loaded test data: {X_test.shape[0]} samples, Image shape = {X_test.shape[1:]}\n")
# LOOP THROUGH MODELS
for model_name, model_path in MODEL_PATHS.items():
    print(f"\n Processing model: {model_name}")
    model = load_model(model_path, compile=False)
    # Predict masks
    y_pred_prob = model.predict(X_test, batch_size=8, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)
    # Create model-specific folder
    model_save_dir = os.path.join(SAVE_DIR, model_name.replace(" ", "_"))
    os.makedirs(model_save_dir, exist_ok=True)
    # Loop through samples
    for idx in tqdm(range(len(X_test)), desc=f"Saving {model_name} visuals"):
        image = X_test[idx]
        gt_mask = y_test[idx].squeeze()
        pred_mask = y_pred[idx].squeeze()
        # Visualization layout
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image[..., :3])  # RGB channels
        plt.title(f"Satellite Image {idx}")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f"{model_name} Prediction")
        plt.axis("off")
        # Save figure
        save_path = os.path.join(model_save_dir, f"sample_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    print(f" Saved results for {model_name} in '{model_save_dir}'")
print("\n All model visualizations generated successfully!")