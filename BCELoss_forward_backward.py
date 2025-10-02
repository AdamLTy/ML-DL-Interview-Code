import numpy as np


"""
def sigmoid(z)

def sigmoid_derivative(z)

class SimpleNN
    def __init__
    def forward
    def backward
    def compute_loss
    def train
    def predict
"""


# Define sigmoid and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Define model and training/inference code
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.weights1) + self.bias1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights2) + self.bias2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, learning_rate):
        # Number of samples
        m = X.shape[0]

        # Calculate output layer error
        dA2 = self.A2 - y
        dZ2 = dA2 * sigmoid_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Calculate hidden layer error
        dA1 = np.dot(dZ2, self.weights2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * db2

    def compute_loss(self, y_hat, y):
        # Calculate cross-entropy loss
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    def train(self, X_train, y_train, epochs, learning_rate):
        # Train the model
        for epoch in range(epochs):
            y_hat = self.forward(X_train)  # Forward propagation
            self.backward(X_train, y_train, learning_rate)  # Backward propagation

            # Print loss every 100 iterations
            if epoch % 100 == 0:
                loss = self.compute_loss(y_hat, y_train)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        # Inference, return class labels
        A2 = self.forward(X)
        return (A2 > 0.5).astype(int)
