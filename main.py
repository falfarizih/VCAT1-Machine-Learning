import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper function to load a batch file
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']  # Image data
        labels = batch[b'labels']  # Class labels
    return data, labels

# === Part (a): Load all training data into Xtr ===
def load_training_data(data_dir):
    X_list = []
    for i in range(1, 6):  # There are 5 training batches
        file = os.path.join(data_dir, f"data_batch_{i}")
        data, _ = load_batch(file)
        X_list.append(data)
    Xtr = np.concatenate(X_list)  # Combine all batches
    return Xtr

# === Part (b): Load all test data into Y ===
def load_test_labels(data_dir):
    file = os.path.join(data_dir, "test_batch")
    _, labels = load_batch(file)
    Y = np.array(labels)
    return Y

# === Part (c): Function to show a 32x32x3 image from a 3072 array ===
def show_image(image_array):
    img = image_array.reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape and rotate axes
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# === Example Usage ===
data_dir = 'cifar-10-batches-py'  # Path to the extracted folder

# Part (a)
Xtr = load_training_data(data_dir)
print(f"Training data shape: {Xtr.shape}")  # Should be (50000, 3072)

# Part (b)
Y = load_test_labels(data_dir)
print(f"Test labels shape: {Y.shape}")  # Should be (10000,)

# Part (c)
# Visualize the first image from the training set
show_image(Xtr[75])
