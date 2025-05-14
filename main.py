import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# 1a DATA LOADING (Preparation Phase) 
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels

# 1a- Load Training Data into Xtr
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

# 1b - Load Test Data into Y
def load_test_data(data_dir):
    file = os.path.join(data_dir, "test_batch")
    data, labels = load_batch(file)
    Xte = np.array(data)
    Yte = np.array(labels)
    return Xte, Yte

# 1c - Visualize a Single Image
def show_image(image_array, label):
    img = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# === ALGORITHM ===
# 2a - Compute K-Nearest-Neighbor with L1 Distance
def predict_knn_l1(Xtr, Ytr, Xte, k):
    predictions = []
    for i in range(len(Xte)):
        distances = np.sum(np.abs(Xtr - Xte[i]), axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Ytr[nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

def predict_knn_l2(Xtr, Ytr, Xte, k):
    predictions = []
    for i in range(len(Xte)):
        distances = np.sqrt(np.sum((Xtr - Xte[i])**2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Ytr[nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_knn_for_multiple_k(Xtr, Ytr, Xte, Yte, k_values, distance='l1'):
    accuracies = []
    for k in k_values:
        print(f"Running KNN with k={k} using {distance} distance...")
        if distance == 'l1':
            preds = predict_knn_l1(Xtr, Ytr, Xte, k)
        else:
            preds = predict_knn_l2(Xtr, Ytr, Xte, k)
        acc = calculate_accuracy(Yte, preds)
        print(f"Accuracy for k={k}: {acc:.4f}")
        accuracies.append(acc)
    return accuracies

def plot_accuracies(k_values, acc_l1, acc_l2):
    plt.plot(k_values, acc_l1, marker='o', label='L1-Distanz')
    plt.plot(k_values, acc_l2, marker='s', label='L2-Distanz')
    plt.xlabel("k")
    plt.ylabel("Genauigkeit")
    plt.title("KNN Genauigkeit (L1 vs. L2)")
    plt.grid(True)
    plt.legend()
    plt.show()

# 2b - Visualize First 10 Test Images with Predicted Labels
def visualize_predictions(Xte, predictions, Yte):
    for i in range(10):
        show_image(Xte[i], f"Predicted: {predictions[i]} | Actual: {Yte[i]}")


# === MAIN EXECUTION ===
data_dir = '/Users/khushirajput/Desktop/VCML/VCAT1-Machine-Learning/Data/cifar-10-batches-py'

# Load data
Xtr, Ytr = load_training_data(data_dir)
Xte, Yte = load_test_data(data_dir)

print(f"Training data shape: {Xtr.shape}")
print(f"Test data shape: {Xte.shape}")

# Show 1 sample
show_image(Xtr[0], Ytr[0])

# Predict first 10 for visual check
predicted_labels = predict_knn_l1(Xtr, Ytr, Xte[:10], k=3)
print("Predicted Labels for First 10 Test Images:", predicted_labels)
print("Actual Labels for First 10 Test Images:   ", Yte[:10])
visualize_predictions(Xte, predicted_labels, Yte)

# === RUN ACCURACY EXPERIMENTS ===
k_values = [3, 5, 7]
Xte_small = Xte[:1000]
Yte_small = Yte[:1000]

acc_l1 = run_knn_for_multiple_k(Xtr, Ytr, Xte_small, Yte_small, k_values, distance='l1')
acc_l2 = run_knn_for_multiple_k(Xtr, Ytr, Xte_small, Yte_small, k_values, distance='l2')

plot_accuracies(k_values, acc_l1, acc_l2)

for i, k in enumerate(k_values):
    print(f"K={k} | L1 Accuracy: {acc_l1[i]:.4f} | L2 Accuracy: {acc_l2[i]:.4f}")