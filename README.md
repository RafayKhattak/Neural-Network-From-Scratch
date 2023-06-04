# Neural Network from Scratch
This repository contains an implementation of a neural network from scratch using only NumPy for numerical computations. The purpose of this project is to provide a basic understanding of neural networks and serve as a learning resource for those interested in the fundamentals of deep learning.
## Features
- Neural network implementation from scratch
- Three-layer network architecture: input, hidden, and output
- ReLU activation function for hidden layers
- Softmax activation function for the output layer
- Gradient descent optimization algorithm for training
- Functions for data preprocessing, parameter initialization, prediction, and accuracy calculation
## Implementation Breakdown
1. init_params()
```
# Initialize parameters of the neural network randomly
def init_params():
    W1 = np.random.rand(50, 784) - 0.5
    b1 = np.random.rand(50, 1) - 0.5
    W2 = np.random.rand(20, 50) - 0.5
    b2 = np.random.rand(20, 1) - 0.5
    W3 = np.random.rand(10, 20) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3
```
- This function initializes the parameters of the neural network randomly.
- W1, b1, W2, b2, W3, and b3 are initialized as randomly generated matrices or vectors using np.random.rand().
- The matrices are subtracted by 0.5 to center the values around zero.
- Finally, the initialized parameters are returned.
2. forward_prop()
```
# Forward propagation in the neural network
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3
```
- This function performs forward propagation in the neural network.
- It takes the weights (W1, W2, W3), biases (b1, b2, b3), and input data (X) as input.
- Z1 = W1.dot(X) + b1 performs the linear transformation of the input using weights W1 and biases b1.
- A1 = ReLU(Z1) applies the ReLU activation function element-wise to the linear outputs.
- Z2 = W2.dot(A1) + b2 performs the linear transformation of the hidden layer outputs using weights W2 and biases b2.
- A2 = ReLU(Z2) applies the ReLU activation function element-wise to the linear outputs.
- Z3 = W3.dot(A2) + b3 performs the linear transformation of the hidden layer outputs using weights W3 and biases b3.
- A3 = softmax(Z3) applies the softmax activation function to the linear outputs to obtain the final output probabilities.
- The function returns the intermediate values Z1, A1, Z2, A2, Z3, and the final output A3.
3. backward_prop()
```
# Backward propagation in the neural network
def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m_train * dZ3.dot(A2.T)
    db3 = 1 / m_train * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3
```
- This function performs backward propagation in the neural network to compute the gradients of the weights and biases.
- It takes the intermediate values Z1, A1, Z2, A2, Z3, the weights W1, W2, W3, input data X, and the true labels Y as input.
- one_hot_Y = one_hot(Y) converts the labels Y into one-hot encoded vectors using the one_hot() function.
- dZ3 = A3 - one_hot_Y computes the derivative of the loss function with respect to Z3.
- dW3 = 1 / m_train * dZ3.dot(A2.T) computes the gradient of the weights W3.
- db3 = 1 / m_train * np.sum(dZ3, axis=1, keepdims=True) computes the gradient of the biases b3.
- dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2) computes the derivative of the loss function with respect to Z2.
- dW2 = 1 / m_train * dZ2.dot(A1.T) computes the gradient of the weights W2.
- db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True) computes the gradient of the biases b2.
- dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) computes the derivative of the loss function with respect to Z1.
- dW1 = 1 / m_train * dZ1.dot(X.T) computes the gradient of the weights W1.
- db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True) computes the gradient of the biases b1.
- The function returns the gradients dW1, db1, dW2, db2, dW3, db3.
4. gradient_descent()
```
# Train the neural network using gradient descent
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            predictions = make_predictions(X, W1, b1, W2, b2, W3, b3)
            accuracy = get_accuracy(predictions, Y)
            print("Iteration:", i, "Accuracy:", accuracy)
    return W1, b1, W2, b2, W3, b3
```
- This function trains the neural network using gradient descent.
- It takes input data X, true labels Y, learning rate, and the number of iterations as input.
- It initializes the parameters using initialize_parameters().
- It then performs the specified number of iterations:
- Calls forward_propagation() to obtain the intermediate and output values.
- Calls backward_propagation() to compute the gradients.
- Calls update_parameters() to update the weights and biases using the gradients.
- Prints the iteration number and the accuracy of predictions every 10 iterations.
- Finally, it returns the updated parameters.
5. make_predictions()
```
# Make predictions for a given set of inputs
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions
```
- This function makes predictions for a given set of inputs.
- It takes input data X and the learned weights and biases as input.
- Calls forward_propagation() to obtain the final output probabilities.
- Calls get_predictions() to convert the probabilities into predicted labels.
- Returns the predicted labels.
6. test_prediction()
```
# Test prediction for a specific index in the training set
def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Label:", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```
- This function tests the prediction for a specific index in the training set.
- Takes an index, the learned weights and biases as input.
- Retrieves the image at the specified index from X_train.
- Calls predict() to make a prediction for the image using the learned parameters.
- Prints the predicted label and the true label.
- Reshapes the image and displays it using matplotlib.
7. ReLU()
```
# ReLU activation function
def ReLU(Z):
    return np.maximum(Z, 0)
```
- Implements the Rectified Linear Unit activation function, which returns the maximum between 0 and the input Z.
8. ReLU_deriv()
```
# Derivative of ReLU activation function
def ReLU_deriv(Z):
    return Z > 0
```
- Calculates the derivative of the ReLU activation function, which returns 1 for positive values of Z and 0 otherwise.
9. softmax()
```
# Softmax activation function
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
```
- Implements the softmax activation function, which calculates the probabilities for each class given the input Z.
10. one_hot()
```
# Convert labels to one-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```
- Converts the labels Y into one-hot encoded vectors using NumPy operations.
11. get_predictions()
```
# Get the predicted labels by selecting the index with the highest probability
def get_predictions(A3):
    return np.argmax(A3, axis=0)
```
- Retrieves the predicted labels from the final output probabilities by selecting the index of the highest probability for each example.
12. get_accuracy()
```
# Calculate the accuracy of the predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
```
- np.sum(predictions == Y): This line compares each element in the predictions array with the corresponding element in the Y array and returns an array of boolean values indicating whether the prediction matches the actual label. np.sum then counts the number of True values, which represents the number of correct predictions.
- Y.size: This line divides the number of correct predictions by the total number of examples in the dataset (Y.size). This calculates the accuracy as a decimal value between 0 and 1.
- The function returns the calculated accuracy.
13. update_params()
```
# Update parameters using gradient descent
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3
```
- This function updates the parameters using gradient descent.
- It takes the weights, biases, gradients, and learning rate as input.
- Updates the weights and biases using the gradients and learning rate.
- Returns the updated weights and biases.



