import numpy as np
import gzip
import struct

def load(filename):
    with gzip.open(filename, "rb") as f:
        _igonred, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        return all_pixels.reshape(n_images, columns, rows)
    
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

X_train = prepend_bias(load("/workspaces/Deep-learning/Pert_6/DataSet/train-images-idx3-ubyte.gz"))
X_test = prepend_bias(load(("/workspaces/Deep-learning/Pert_6/DataSet/t10k-images-idx3-ubyte.gz")))

def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        f.read(8)
        all_labels =  f.read()
        return np.frombuffer(all_labels, dtype= np.uint8).reshape(-1, 1)
    
def one_hot_encoded(Y):
    n_lables = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_lables, n_classes))
    for i in range(n_lables):
        label = Y[i]
        encoded_Y[i][label] = 1

Y_train = one_hot_encoded(load_labels("/workspaces/Deep-learning/Pert_6/DataSet/train-labels-idx1-ubyte.gz"))
Y_test = one_hot_encoded(load_labels("/workspaces/Deep-learning/Pert_6/DataSet/t10k-labels-idx1-ubyte.gz"))