#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gzip

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = [int.from_bytes(f.read(4), 'big') for _ in range(4)]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        images = images / 255.0
        return images

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)


def one_hot_encode_tanh(labels, num_classes=10):
    one_hot = -np.ones((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def one_hot_encode_relu(labels, num_classes=10):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

class Neural_Network_tanh:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def tanh(self, x):
        return np.tanh(x)

    def derive_tanh(self, x):
        return 1 - np.tanh(x)**2

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)
        return self.a2

    def mse(self, Y):
        self.error = self.a2 - Y
        return -np.mean(np.power(self.error, 2))


    def backpropagation(self, X, Y, learning_rate):
        m = Y.shape[0]

        derivative_z2 = self.a2 - Y  # Error derivative
        dW2 = np.dot(self.a1.T, derivative_z2) / m  # Gradient for W2
        db2 = np.sum(derivative_z2, axis=0, keepdims=True) / m  # Gradient for b2

        derivative_z1 = np.dot(derivative_z2, self.W2.T) * self.derive_tanh(self.z1)
        dW1 = np.dot(X.T, derivative_z1) / m  # Gradient for W1
        db1 = np.sum(derivative_z1, axis=0, keepdims=True) / m  # Gradient for b1

        # Update the weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=1)


    def train(self, X, Y, epochs, learning_rate):
        m = X.shape[0]

        try:
            for epoch in range(epochs):
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                self.feedforward(X_shuffled)
                loss = self.mse(Y_shuffled)
                self.backpropagation(X_shuffled, Y_shuffled, learning_rate)

                if epoch % 10 == 0:
                    predictions = self.predict(X_shuffled)
                    accuracy = np.mean(predictions == np.argmax(Y_shuffled, axis=1))
                    print(f'Epoch {epoch}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Proceeding to test set evaluation.")

    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))

        incorrect_indices = np.where(predictions != np.argmax(Y_test, axis=1))[0]
        incorrect_patterns = [(index, predictions[index], np.argmax(Y_test[index])) for index in incorrect_indices]

        return accuracy, incorrect_patterns
    
    

    

class Neural_Network_ReLU:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def derive_relu(self, x):
        return x > 0

    def sigmoid(self, x):
        # Improved for numerical stability
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def mse(self, Y):
        self.error = self.a2 - Y
        return -np.mean(np.power(self.error, 2))


    def backpropagation(self, X, Y, learning_rate):
        m = Y.shape[0]

        derivative_z2 = self.a2 - Y
        dW2 = np.dot(self.a1.T, derivative_z2) / m
        db2 = np.sum(derivative_z2, axis=0, keepdims=True) / m

        derivative_z1 = np.dot(derivative_z2, self.W2.T) * self.derive_relu(self.z1)
        dW1 = np.dot(X.T, derivative_z1) / m
        db1 = np.sum(derivative_z1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=1)


    def train(self, X, Y, epochs, learning_rate):
        m = X.shape[0]

        try:
            for epoch in range(epochs):
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                self.feedforward(X_shuffled)
                loss = self.mse(Y_shuffled)
                self.backpropagation(X_shuffled, Y_shuffled, learning_rate)

                if epoch % 10 == 0:
                    predictions = self.predict(X_shuffled)
                    accuracy = np.mean(predictions == np.argmax(Y_shuffled, axis=1))
                    print(f'Epoch {epoch}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Proceeding to test set evaluation.")

    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))

        incorrect_indices = np.where(predictions != np.argmax(Y_test, axis=1))[0]
        incorrect_patterns = [(index, predictions[index], np.argmax(Y_test[index])) for index in incorrect_indices]

        return accuracy, incorrect_patterns
    
    
def write_results_to_file_firstrun(filename, activation_type, hidden_size, learning_rate, accuracy, incorrect_patterns):
    with open(filename, 'a') as file:
        file.write(f"Activation: {activation_type}, Hidden Layer Size: {hidden_size}, Learning Rate: {learning_rate}\n")
        file.write(f"Test Accuracy: {100*accuracy:.2f}%\n")
        file.write("Incorrect Patterns:\n")
        for index, predicted, actual in incorrect_patterns:
            file.write(f"Index: {index}, Predicted: {predicted}, Actual: {actual}\n")
        file.write("\n")

        



        
class Neural_Network_tanh_with_batch:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def tanh(self, x):
        return np.tanh(x)

    def derive_tanh(self, x):
        return 1 - np.tanh(x)**2

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)
        return self.a2

    def mse(self, Y):
        self.error = self.a2 - Y
        return -np.mean(np.power(self.error, 2))


    def backpropagation(self, X, Y, learning_rate):
        m = Y.shape[0]

        derivative_z2 = self.a2 - Y  # Error derivative
        dW2 = np.dot(self.a1.T, derivative_z2) / m  # Gradient for W2
        db2 = np.sum(derivative_z2, axis=0, keepdims=True) / m  # Gradient for b2

        derivative_z1 = np.dot(derivative_z2, self.W2.T) * self.derive_tanh(self.z1)
        dW1 = np.dot(X.T, derivative_z1) / m  # Gradient for W1
        db1 = np.sum(derivative_z1, axis=0, keepdims=True) / m  # Gradient for b1

        # Update the weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=1)


    def train_mini_batch(self, X, Y, epochs, learning_rate, batch_size):
        m = X.shape[0]

        try:
            for epoch in range(epochs):
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                # Mini-batch training
                for i in range(0, m, batch_size):
                    X_mini = X_shuffled[i:i + batch_size]
                    Y_mini = Y_shuffled[i:i + batch_size]
                    self.feedforward(X_mini)
                    self.backpropagation(X_mini, Y_mini, learning_rate)

                output = self.feedforward(X_shuffled)
                loss = self.mse(Y_shuffled)

                if epoch % 10 == 0:
                    predictions = self.predict(X_shuffled)
                    accuracy = np.mean(predictions == np.argmax(Y_shuffled, axis=1))
                    print(f'Epoch {epoch}, Loss: {loss:.7f}, Accuracy: {accuracy:.7f}')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Proceeding to test set evaluation.")

    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))

        incorrect_indices = np.where(predictions != np.argmax(Y_test, axis=1))[0]
        incorrect_patterns = [(index, predictions[index], np.argmax(Y_test[index])) for index in incorrect_indices]

        return accuracy, incorrect_patterns



def write_results_to_file_secondrun(filename, batch_size, accuracy, incorrect_patterns):
    with open(filename, 'a') as file:
        file.write(f"Batch Size: {batch_size}\n")
        file.write(f"Test Accuracy: {100*accuracy:.5f}%\n")
        file.write("Incorrect Patterns:\n")
        for index, predicted, actual in incorrect_patterns:
            file.write(f"Index: {index}, Predicted: {predicted}, Actual: {actual}\n")
        file.write("\n")
        
        

        
        
        
class NN_tanh_batch_regularization:
    def __init__(self, input_size, hidden_size, output_size, lambda_):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lambda_ = lambda_
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def tanh(self, x):
        return np.tanh(x)

    def derive_tanh(self, x):
        return 1 - np.tanh(x)**2

    def feedforward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1) 
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2) 
        return self.a2

    def mse(self, Y):
        self.error = self.a2 - Y
        data_loss = np.mean(np.power(self.error, 2))
        reg_loss = (self.lambda_ / (2 * Y.shape[0])) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        total_loss = data_loss + reg_loss
        return total_loss


    def backpropagation(self, X, Y, learning_rate):
        m = Y.shape[0]

        derivative_z2 = self.a2 - Y  # Error derivative
        dW2 = np.dot(self.a1.T, derivative_z2) / m  # Gradient for W2
        db2 = np.sum(derivative_z2, axis=0, keepdims=True) / m  # Gradient for b2

        derivative_z1 = np.dot(derivative_z2, self.W2.T) * self.derive_tanh(self.z1)
        dW1 = np.dot(X.T, derivative_z1) / m  # Gradient for W1
        db1 = np.sum(derivative_z1, axis=0, keepdims=True) / m  # Gradient for b1

        # Update the weights and biases
        self.W1 -= learning_rate * (dW1 + (self.lambda_ / m) * self.W1)
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * (dW2 + (self.lambda_ / m) * self.W2)
        self.b2 -= learning_rate * db2

    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=1)


    def train_regularization(self, X, Y, epochs, learning_rate, batch_size):
        m = X.shape[0]

        try:
            for epoch in range(epochs):
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                # Mini-batch training
                for i in range(0, m, batch_size):
                    X_mini = X_shuffled[i:i + batch_size]
                    Y_mini = Y_shuffled[i:i + batch_size]
                    self.feedforward(X_mini)
                    self.backpropagation(X_mini, Y_mini, learning_rate)

                output = self.feedforward(X_shuffled)
                loss = self.mse(Y_shuffled)  # Regularized loss is calculated here

                if epoch % 10 == 0:
                    predictions = self.predict(X_shuffled)
                    accuracy = np.mean(predictions == np.argmax(Y_shuffled, axis=1))
                    print(f'Epoch {epoch}, Loss: {loss:.7f}, Accuracy: {accuracy:.7f}')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Proceeding to test set evaluation.")


    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))

        incorrect_indices = np.where(predictions != np.argmax(Y_test, axis=1))[0]
        incorrect_patterns = [(index, predictions[index], np.argmax(Y_test[index])) for index in incorrect_indices]

        return accuracy, incorrect_patterns



def write_results_to_file_regularization(filename, batch_size, accuracy, incorrect_patterns):
        with open(filename, 'a') as file:
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Test Accuracy: {100*accuracy:.5f}%\n")
            file.write("Incorrect Patterns:\n")
            for index, predicted, actual in incorrect_patterns:
                file.write(f"Index: {index}, Predicted: {predicted}, Actual: {actual}\n")
            file.write("\n")
            
            
def main():
    train_images = load_images('train-images-idx3-ubyte.gz')
    train_labels = load_labels('train-labels-idx1-ubyte.gz')

    tanh_train_labels = one_hot_encode_tanh(train_labels)
    relu_train_labels = one_hot_encode_relu(train_labels)

    test_labels = load_labels('t10k-labels-idx1-ubyte.gz')
    test_images = load_images('t10k-images-idx3-ubyte.gz')

    tanh_test_labels = one_hot_encode_tanh(test_labels)
    relu_test_labels = one_hot_encode_relu(test_labels)
    
    print("To skip the training process, press Ctrl+C.")
    print("Warning: To re-write result files, files should be deleted before each run.")

    hidden_layer_sizes = [300, 500, 1000]
    learning_rates = [0.01, 0.05, 0.09]

    # ReLU Training
    for hidden_size in hidden_layer_sizes:
        for learning_rate in learning_rates:
            nn = Neural_Network_ReLU(input_size=784, hidden_size=hidden_size, output_size=10)
            nn.train(train_images, relu_train_labels, epochs=100, learning_rate=learning_rate)
            accuracy, incorrect_patterns = nn.evaluate(test_images, relu_test_labels)
            print(f'Test Accuracy: {100*accuracy:.2f}% for Hidden Layer Size: {hidden_size}, Learning Rate: {learning_rate}, Activation: ReLU & sigmoid\n')
            write_results_to_file_firstrun("relu_results.txt", "ReLU & sigmoid", hidden_size, learning_rate, accuracy, incorrect_patterns)

    # tanh Training
    for hidden_size in hidden_layer_sizes:
        for learning_rate in learning_rates:
            nn = Neural_Network_tanh(input_size=784, hidden_size=hidden_size, output_size=10)
            nn.train(train_images, tanh_train_labels, epochs=100, learning_rate=learning_rate)
            accuracy, incorrect_patterns = nn.evaluate(test_images, tanh_test_labels)
            print(f'Test Accuracy: {100*accuracy:.2f}% for Hidden Layer Size: {hidden_size}, Learning Rate: {learning_rate}, Activation: tanh\n')
            write_results_to_file_firstrun("tanh_results.txt", "tanh", hidden_size, learning_rate, accuracy, incorrect_patterns)

    best_network = Neural_Network_tanh_with_batch(input_size=784, hidden_size=1000, output_size=10)

    # Mini-Batch Training with different batch sizes
    batch_sizes = [10, 50, 100]
    for batch_size in batch_sizes:
        print(f'\nTraining with batch size: {batch_size}')
        best_network.train_mini_batch(train_images, tanh_train_labels, epochs=30, learning_rate=0.09, batch_size=batch_size)
        accuracy, incorrect_patterns = best_network.evaluate(test_images, tanh_test_labels)
        print(f'Accuracy for batch size {batch_size}: {100*accuracy}')
        write_results_to_file_secondrun("tanh_with_batch_results.txt", batch_size, accuracy, incorrect_patterns)

    # L2 Regularization Training with different lambda values
    lambdas = [0.001, 0.01]
    for lambda_val in lambdas:
        best_network = NN_tanh_batch_regularization(input_size=784, hidden_size=1000, output_size=10, lambda_=lambda_val)
        print(f'\nTraining with lambda: {lambda_val}')
        best_network.train_regularization(train_images, tanh_train_labels, epochs=50, learning_rate=0.09, batch_size=100)
        accuracy, incorrect_patterns = best_network.evaluate(test_images, tanh_test_labels)
        print(f'Accuracy for lambda {lambda_val}: {100*accuracy:.2f}%')
        write_results_to_file_regularization(f"tanh_with_batch_reg_results_lambda_{lambda_val}.txt", 100, accuracy, incorrect_patterns)

if __name__ == "__main__":
    main()




