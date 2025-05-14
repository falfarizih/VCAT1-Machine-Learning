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
def predict_knn(Xtr, Ytr, Xte, k=7):
    predictions = []
    for i in range(len(Xte)):
        distances = np.sum(np.abs(Xtr - Xte[i]), axis=1)
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Ytr[nearest_indices]

        label_counts = Counter(nearest_labels)
        most_common = label_counts.most_common()

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie case: pick the label of the nearest neighbor
            prediction = Ytr[nearest_indices[0]]
        else:
            prediction = most_common[0][0]

        predictions.append(prediction)
    return np.array(predictions)

# Part (b) - Visualize First 10 Test Images with Predicted Labels
def visualize_predictions(Xte, predictions):
    for i in range(10):
        show_image(Xte[i], predictions[i])
# Part (d): Calculate accuracy
def calculate_accuracy(predictions, yte):
    correct = np.sum(predictions == yte)
    return correct / len(yte)

# Part (d): Plot accuracies for K values
def plot_accuracies(k_values, accuracies):
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('K-NN Accuracy for Different K Values')
    plt.grid(True)
    plt.show()

# === PART (e): New Functions for L2 Distance ===

# K-NN Predictor Using L2 Distance
def predict_knn_l2(xtr, ytr, xte, k=3):
    predictions = []
    for i in range(len(xte)):
        if i % 100 == 0:
            print(f"[L2] Processing test image {i}/{len(xte)}...")

        distances = np.sqrt(np.sum((xtr - xte[i]) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = ytr[nearest_indices]

        label_counts = Counter(nearest_labels)
        most_common = label_counts.most_common()

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            prediction = ytr[nearest_indices[0]]  # Tie-break using smallest distance
        else:
            prediction = most_common[0][0]

        predictions.append(prediction)
    return np.array(predictions)

# Accuracy Calculation for L2 Distance (Reuse Existing calculate_accuracy)

# Visualization and Evaluation for L2 Distance
def evaluate_knn_l2(xtr, ytr, xte, yte):
    k_values = [1, 3, 5, 7]
    accuracies = []

    print("\nAccuracy Results Using L2 Distance")
    for k in k_values:
        preds = predict_knn_l2(xtr, ytr, xte, k=k)
        acc = calculate_accuracy(preds, yte)
        accuracies.append(acc)
        print(f"[L2] Accuracy for K={k}: {acc * 100:.2f}%")

    plot_accuracies(k_values, accuracies)

    best_k = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies) * 100
    print(f"\n✅ [L2] Best K is {best_k} with Accuracy: {best_acc:.2f}%")
    return best_acc

# === MAIN EXECUTION ===
data_dir = 'cifar-10-batches-py'  # Path to your extracted CIFAR-10 folder

# Preparation Phase
Xtr, Ytr = load_training_data(data_dir)  # Part (a)
print(f"Training data shape (Xtr): {Xtr.shape}")  # Should be (50000, 3072)

Xte, Yte = load_test_data(data_dir)  # Part (b)
print(f"Test data shape (Xte): {Xte.shape}")     # Should be (10000,)

# Part (c) - Visualize the first training image
show_image(Xtr[0], Ytr[0])

test_subset = Xte[:50]
test_labels = Yte[:50]

k_values_d = [1, 3, 5, 7]
accuracies = []

print("\nAccuracy Results")
for k in k_values_d:
    preds = predict_knn(Xtr, Ytr, test_subset, k=k)
    acc = calculate_accuracy(preds, test_labels)
    accuracies.append(acc)
    print(f"Accuracy for K={k}: {acc * 100:.2f}%")

# Plot the accuracy results
plot_accuracies(k_values_d, accuracies)

# Determine and explicitly show the best K value
best_k = k_values_d[np.argmax(accuracies)]
best_acc = max(accuracies) * 100
print(f"\n✅ Best K value is {best_k} with Accuracy: {best_acc:.2f}%")
#part e
best_l2_accuracy = evaluate_knn_l2(Xtr, Ytr, test_subset, test_labels)

# Algorithm Phase
predicted_labels = predict_knn(Xtr, Ytr, Xte[:10], k=7)  # Predict for first 10 test samples

# Explicitly print results for professor
print("Predicted Labels for First 10 Test Images:", predicted_labels)
print("Actual Labels for First 10 Test Images:   ", Yte[:10])

# Visualize Predictions
visualize_predictions(Xte, predicted_labels)
