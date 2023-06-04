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
This function makes predictions for a given set of inputs.
It takes input data X and the learned weights and biases as input.
Calls forward_propagation() to obtain the final output probabilities.
Calls get_predictions() to convert the probabilities into predicted labels.
Returns the predicted labels.
