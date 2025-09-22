import numpy as np
def load_data(data_path="data/"):
    X_train = np.load(f"{data_path}/trainX.npy")
    y_train = np.load(f"{data_path}/trainY.npy")
    X_val   = np.load(f"{data_path}/valX.npy")
    y_val   = np.load(f"{data_path}/valY.npy")
    X_test  = np.load(f"{data_path}/testX.npy")
    y_test  = np.load(f"{data_path}/testY.npy")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)