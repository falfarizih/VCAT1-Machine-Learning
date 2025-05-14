import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# Data Loading
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels

# Load Training Data into Xtr
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

# Load Test Data into Y
def load_test_data(data_dir):
    file = os.path.join(data_dir, "test_batch")
    data, labels = load_batch(file)
    Xte = np.array(data)
    Yte = np.array(labels)
    return Xte, Yte

# Visualize a Single Image
def show_image(image_array, label):
    img = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# Compute K-Nearest-Neighborwith chosen distance type
def predict_knn(Xtr, Ytr, Xte, k=3, distance_type='l1'):
    predictions = []
    for i in range(len(Xte)):
        if distance_type == 'l1':
            distances = np.sum(np.abs(Xtr - Xte[i]), axis=1)
        elif distance_type == 'l2':
            distances = np.sqrt(np.sum((Xtr - Xte[i]) ** 2, axis=1))
        else:
            raise ValueError("Invalid distance_type.")

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Ytr[nearest_indices]

        label_counts = Counter(nearest_labels)
        most_common = label_counts.most_common()

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            prediction = Ytr[nearest_indices[0]]  # Tie-breaker: nearest neighbor
        else:
            prediction = most_common[0][0]

        predictions.append(prediction)
    return np.array(predictions)

# VisualizeTest Images with Predicted Labels
def visualize_predictions(Xte, predictions):
    for i in range(10):
        show_image(Xte[i], predictions[i])

# Calculate accuracy
def calculate_accuracy(predictions, yte):
    correct = np.sum(predictions == yte)
    return correct / len(yte)

# Plot accuracies for K values
def plot_accuracies(k_values, accuracies, distance_type):
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title(f'K-NN Accuracy ({distance_type.upper()} Distance)')
    plt.grid(True)
    plt.show()

# main
data_dir = 'cifar-10-batches-py'  # Path CIFAR-10 folder

# Preparation Phase
Xtr, Ytr = load_training_data(data_dir)
print(f"\nTraining data shape (Xtr): {Xtr.shape}")  # Should be (50000, 3072)

Xte, Yte = load_test_data(data_dir)  # Part (b)
print(f"Test data shape (Xte): {Xte.shape}")     # Should be (10000,3072)

print("\nPredicting first 10 test images using L1 Distance, K=7...")
predicted_labels = predict_knn(Xtr, Ytr, Xte[:10], k=7, distance_type='l1')

print("\nPredicted Labels for First 10 Test Images:", predicted_labels)
print("Actual Labels for First 10 Test Images:   ", Yte[:10])
# Visualize the first training image
show_image(Xtr[0], Ytr[0])

# Evaluation Phase
test_subset = Xte[:1000]
test_labels = Yte[:1000]

k_values = [1, 3, 5, 7]
distance_metrics = ['l1', 'l2']

for distance in distance_metrics:
    print(f"\n{distance.upper()} Distance Results:")
    accuracies = []
    for k in k_values:
        preds = predict_knn(Xtr, Ytr, test_subset, k=k, distance_type=distance)
        acc = calculate_accuracy(preds, test_labels)
        accuracies.append(acc)
        print(f"Accuracy for K={k}: {acc * 100:.2f}%")

    plot_accuracies(k_values, accuracies, distance)

    best_k = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies) * 100
    print(f"Best K: {best_k} with Accuracy: {best_acc:.2f}%")


visualize_predictions(Xte, predicted_labels)