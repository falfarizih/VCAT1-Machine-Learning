import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# [1.a] Load all training batches into variable Xtr
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    X = data[b'data']  # Image data
    Y = data[b'labels']  # Corresponding labels
    return X, Y

def load_training_data(data_folder):
    X_train = []
    Y_train = []
    for i in range(1, 6):  # There are 5 training batches (data_batch_1 to data_batch_5)
        batch_file = os.path.join(data_folder, f'data_batch_{i}')
        X_batch, Y_batch = load_batch(batch_file)
        X_train.append(X_batch)
        Y_train.extend(Y_batch)
    X_train = np.vstack(X_train)  # Stack all batches vertically
    Y_train = np.array(Y_train)
    return X_train, Y_train  # Xtr, Ytr

# [1.b] Load test batch into variable Y
def load_test_data(data_folder):
    X_test, Y_test = load_batch(os.path.join(data_folder, 'test_batch'))
    return np.array(X_test), np.array(Y_test)

# Example usage for loading data
data_folder = 'cifar-10-batches-py'  # Change this if your path is different
Xtr, Ytr = load_training_data(data_folder)  # Training data and labels
X_test, Y_test = load_test_data(data_folder)  # Test data and labels

print("Training data shape (Xtr):", Xtr.shape)
print("Test data shape (X_test):", X_test.shape)
