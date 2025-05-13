import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# === DATA LOADING (Preparation Phase) ===
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels

# Part (a) - Load Training Data into Xtr
def load_training_data(data_dir):
    x_list, y_list = [], []
    for i in range(1, 6):
        file = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_batch(file)
        x_list.append(data)
        y_list.extend(labels)
    Xtr = np.concatenate(x_list)
    Ytr = np.array(y_list)
    return Xtr, Ytr

# Part (b) - Load Test Data into Y
def load_test_data(data_dir):
    file = os.path.join(data_dir, "test_batch")
    data, labels = load_batch(file)
    Xte = np.array(data)
    Yte = np.array(labels)
    return Xte, Yte

# Part (c) - Visualize a Single Image
def show_image(image_array, label):
    img = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# === ALGORITHM ===
# Part (a) - Compute K-Nearest-Neighbor with L1 Distance
def predict_knn(Xtr, Ytr, Xte, k=3):
    predictions = []
    for i in range(len(Xte)):
        distances = np.sum(np.abs(Xtr - Xte[i]), axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Ytr[nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

# Part (b) - Visualize First 10 Test Images with Predicted Labels
def visualize_predictions(Xte, predictions):
    for i in range(10):
        predicted_label = predictions[i]
        actual_label = Yte[i]
        show_image(Xte[i], f"Predicted: {predicted_label} | Actual: {actual_label}")

# === MAIN EXECUTION ===
data_dir = 'cifar-10-batches-py'  # Path to your extracted CIFAR-10 folder

# Preparation Phase
Xtr, Ytr = load_training_data(data_dir)  # Part (a)
print(f"Training data shape (Xtr): {Xtr.shape}")  # Should be (50000, 3072)

Xte, Yte = load_test_data(data_dir)  # Part (b)
print(f"Test data shape (Xte): {Xte.shape}")     # Should be (10000,)

# Part (c) - Visualize the first training image
show_image(Xtr[0], Ytr[0])

# Algorithm Phase
predicted_labels = predict_knn(Xtr, Ytr, Xte[:10], k=3)  # Predict for first 10 test samples

# Explicitly print results for professor
print("Predicted Labels for First 10 Test Images:", predicted_labels)
print("Actual Labels for First 10 Test Images:   ", Yte[:10])

# Visualize Predictions
visualize_predictions(Xte, predicted_labels)
