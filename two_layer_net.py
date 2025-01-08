import numpy as np
import matplotlib.pyplot as plt
import os

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Forward pass: compute class scores
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        # If the targets are not given then return the scores
        if y is None:
            return scores

        # Compute the loss: Softmax loss
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        softmax_matrix = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=1, keepdims=True)
        correct_class_scores = softmax_matrix[np.arange(N), y]
        loss = -np.sum(np.log(correct_class_scores)) / N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backward pass: compute gradients
        grads = {}

        softmax_matrix[np.arange(N), y] -= 1
        softmax_matrix /= N
        grads['W2'] = np.dot(hidden_layer.T, softmax_matrix) + 2 * reg * W2
        grads['b2'] = np.sum(softmax_matrix, axis=0)

        hidden_grad = np.dot(softmax_matrix, W2.T)
        hidden_grad[hidden_layer <= 0] = 0

        grads['W1'] = np.dot(X.T, hidden_grad) + 2 * reg * W1
        grads['b1'] = np.sum(hidden_grad, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # Update the parameters using the gradients
            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            if verbose and it % 100 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss}')

            # Every epoch, check train and val accuracy and decay learning rate
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
        """
        hidden_layer = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
        scores = np.dot(hidden_layer, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred


plt.rcParams['figure.figsize'] = (10.0, 8.0)  # Set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """Returns relative error."""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Create a small network and toy data to check implementations
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

# Initialize model and data
net = init_toy_model()
X, y = init_toy_data()

# Forward pass: Compute the scores
scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('Correct scores:')
correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))

# Compute the loss
loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# The difference should be very small
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

# Plot the loss history
train_results = net.train(X, y, X, y, num_iters=1000, verbose=False)
plt.plot(train_results['loss_history'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.savefig('templates/TLn_training_loss.png')
plt.show()
