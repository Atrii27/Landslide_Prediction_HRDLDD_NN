#  Landslide Detection using Deep Learning (HRDLDD)
This project implements multiple deep learning architectures for **satellite-based landslide detection** using semantic segmentation.  
It compares U-Net, ResU-Net, Attention U-Net, Attention ResU-Net, and ADSMS U-Net models on Landslide Satellite imagery.
##  Directory Structure
Landslide_Detection_HRDLDD/
│
├── data/
│   ├── test/
│   │   ├── testX.npy
│   │   └── testY.npy
│   │
│   ├── train/
│   │   ├── trainX.npy
│   │   └── trainY.npy
│   │
│   └── val/
│       ├── valX.npy
│       └── valY.npy
│
├── models/
│   ├── unet.py
│   ├── resnet.py
│   ├── attn_unet.py
│   ├── attn_resunet.py
│   └── adsms_unet.py
│
├── result images/
│   ├── U_Net/
│   │   └── (354 predicted result images)
│   │
│   ├── ResU-Net/
│   │   └── (354 predicted result images)
│   │
│   ├── Attn_U-Net/
│   │   └── (354 predicted result images)
│   │
│   ├── Attn_ResU-Net/
│   │   └── (354 predicted result images)
│   │
│   └── ADSMS_U-Net/
│       └── (354 predicted result images)
│
├── saved_models/
│   ├── unet.h5
│   ├── resunet_model.h5
│   ├── attnunet_model.h5
│   ├── attn_resunet.h5
│   └── adsms_unet.h5
│
├── utils.py
├── looses.py
├── train_all.py
├── evaluate.py
├── visuals.py
├── classification_visuals.py
├── model_comparision.png
├── result_comparision.png
├── Results.txt
└── y_pred.npy
## Models Implemented
| Model Name        | File | Key Features |
|-------------------|------|---------------|
| **U-Net** | `models/unet.py` | Baseline encoder-decoder architecture |
| **ResU-Net** | `models/resnet.py` | Residual skip connections |
| **Attention U-Net** | `models/attn_unet.py` | Attention gates for better focus on landslide regions |
| **Attention ResU-Net** | `models/attn_resunet.py` | Combines residual and attention mechanisms |
| **ADSMS U-Net** | `models/adsms_unet.py` | Multi-scale attention-based enhanced segmentation |
## Evaluation Metrics
| Model | Accuracy | F1 Score |
|--------|-----------|-----------|
| **U-Net** | 0.9479 | 0.7237 |
| **ResU-Net** | 0.9359 | 0.6739 |
| **Attention U-Net** | 0.9397 | 0.7357 |
| **Attention ResU-Net** | 0.9375 | 0.6573 |
| **ADSMS U-Net** | 0.9401 | 0.6743 |
Results are visualized in:
- `model_comparision.png`  
- `result_comparision.png`
## Visualization
Use `classification_visuals.py` and `visuals.py`to generate side-by-side comparisons:
- Satellite Image  
- Ground Truth Mask  
- Predicted Mask  
## How to Run
### Install dependencies
pip install -r requirements.txt
python train_all.py
python evaluate.py
python classification_visuals.py
Requirements-
Python 3.10+
TensorFlow / Keras for model building
NumPy, Matplotlib, Seaborn for analysis
Scikit-learn for metrics
## Acknowledgement
Developed as part of an internship under Dr. Yunis Ali Pulapdan, focusing on high-resolution landslide detection through deep learning.


