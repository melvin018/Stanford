import numpy as np
import matplotlib.pyplot as plt
import data_utils
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the raw CIFAR-10 data
cifar10_dir = './data/cifar-10-batches-py'
X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

# Subsample the data to use only 500 training examples and 250 test examples
num_training = 500
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 250
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Initialize k-NN classifier
k_values = [1, 3, 5, 7, 9]  # Values of k to try
accuracies = []

for k in k_values:
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy of k-NN classifier with k={k} on CIFAR-10 test set: {accuracy * 100:.2f}%")

# Save the plot under a folder named "templates" with an appropriate name
if not os.path.exists('templates'):
    os.makedirs('templates')

file_path = os.path.join('templates', 'cifar10_knn_accuracy.png')

plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(file_path)

print(f"Plot saved successfully at {file_path}")
