# Landslide Detection using Deep Learning (HRDLDD)
This repository implements **U-Net, ResU-Net, and Attention U-Net** for landslide detection using the **HR-GLDD** dataset (converted into `.npy` format).  
The models perform semantic segmentation to identify landslide areas from satellite images.
##  Repository Structure
- `data/` → Numpy datasets (`trainX.npy`, `trainY.npy`, etc.)
- `models/` → Architectures (`unet.py`, `resnet.py`, `attennet.py`)
- `saved_models/` → Trained `.h5` models
- `train.py` → Train a chosen model
- `evaluate.py` → Evaluate models (Precision, Recall, F1, plots)
- `utils.py` → Data loading utilities
- `Results.txt` → Logs of model metrics
- `losses.txt` → Training losses
- `notebooks/` → Colab/Jupyter notebooks for experiments
## Setup
Data Set - https://zenodo.org/records/7189381
git clone https://github.com/YOUR_USERNAME/landslide-detection-HRDLDD.git
cd landslide-detection-HRDLDD
pip install -r requirements.txt
## Training 
# Train U-Net 
python train.py --model unet --epochs 50
# Train ResU-Net
python train.py --model resnet --epochs 50
# Train Attention U-Net
python train.py --model attennet --epochs 50
## Evaluation 
python evaluate.py
## Visualization
python evaluate.py
python plot_results.py
## References
U-Net: Convolutional Networks for Biomedical Image Segmentation
ResUNet: A Deep Residual U-Net for Image Segmentation
Attention U-Net: Learning Where to Look for the Pancreas
HR-GLDD Dataset Paper


