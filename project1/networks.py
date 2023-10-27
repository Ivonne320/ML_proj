import numpy as np
from implementations import *
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# def initialize_network(input_size, hidden_size, output_size):
def initialize_network(input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
    network = []

    hidden_layer1 = {'W': np.random.randn(input_size, hidden_size1)*np.sqrt(2./input_size),
                    'b': np.zeros((1, hidden_size1))}
    hidden_layer2 = {'W': np.random.randn(hidden_size1, hidden_size2)*np.sqrt(2./hidden_size1),
                    'b': np.zeros((1, hidden_size2))}
    hidden_layer3 = {'W': np.random.randn(hidden_size2, hidden_size3)*np.sqrt(2./hidden_size2),
                    'b': np.zeros((1, hidden_size3))}
    output_layer = {'W': np.random.randn(hidden_size3, output_size)*np.sqrt(2./hidden_size3),
                    'b': np.zeros((1, output_size))}
    network.append(hidden_layer1)
    network.append(hidden_layer2)
    network.append(hidden_layer3)
    network.append(output_layer)

    return network

def forward_propagation(network, x):
    z1 = np.dot(x, network[0]['W']) + network[0]['b']
    a1 = leaky_relu(z1)
   
    z2 = np.dot(a1, network[1]['W']) + network[1]['b'] 
    a2 = leaky_relu(z2)

    z3 = np.dot(a2, network[2]['W']) + network[2]['b']
    a3 = leaky_relu(z3)


    z4 = np.dot(a3, network[3]['W']) + network[3]['b']
    a4 = sigmoid(z4)

    activations = [a1, a2, a3, a4]
    return activations

def backpropagation(network, x, y, activations):
    m = x.shape[0]
    # a1, a2 = activations
    a1, a2, a3, a4 = activations
    

    dz4 = a4 - y.reshape(-1, 1)
    dw4 = np.dot(a3.T, dz4) / m
    db4 = np.sum(dz4, axis=0, keepdims=True) / m

    da3 = np.dot(dz4, network[3]['W'].T)
    dz3 = da3 * leaky_relu_derivative(a3)
    dw3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m

    da2 = np.dot(dz3, network[2]['W'].T)
    dz2 = da2 * leaky_relu_derivative(a2)
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, network[1]['W'].T)
    dz1 = da1 * leaky_relu_derivative(a1)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    max_int = 1

    # return [{"dW": dw1, "db": db1}, {"dW": dw2, "db": db2}]
    return [{"dW": dw1, "db": db1}, {"dW": dw2, "db": db2}, {"dW": dw3, "db": db3}, {"dW": dw4, "db": db4}]

def update_weights(network, gradients, learning_rate):
    for layer, gradient in zip(network, gradients):
        layer['W'] -= learning_rate * gradient['dW']
        layer['b'] -= learning_rate * gradient['db']

def train(network, X, y, X_v, y_v, learning_rate, epochs, batch_size, n_pat=10):
    m = X.shape[0]
    losses = []
    f1s = []
    accs = []
    best_f1 = -np.inf
    num_epochs_without_improvement = 0
    patience = n_pat
    for epoch in range(epochs):
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            activations = forward_propagation(network, X_batch)
            # a2 = activations[-1]
            # loss = -np.mean(y_batch * np.log(a2) + (1 - y_batch) * np.log(1 - a2))
            a4 = activations[-1]
            loss = -np.mean(y_batch.reshape(-1, 1) * np.log(a4) + (1 - y_batch.reshape(-1, 1)) * np.log(1 - a4))
            losses.append(loss)
            # logprobs = np.dot(y_batch.reshape(1,-1),np.log(a4))+np.dot((1-y_batch.reshape(1,-1)),np.log(1-a4)) 
            # loss = -logprobs / m
        
            gradients = backpropagation(network, X_batch, y_batch, activations)
            update_weights(network, gradients, learning_rate)
        # test the model on validation set
        y_pred = predict(network, X_v)
        f1s.append(predict_f1_pure(y_pred, y_v))
        accs.append(predict_acc_pure(y_pred, y_v))
        # early stopping
        if f1s[-1] > best_f1:
            best_f1 = f1s[-1]
            num_epochs_without_improvement = 0
            best_network = network.copy()
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement > patience:
                print("Early stopping at epoch", epoch)
                network = best_network
                break
        # print index every 2 epochs
        if epoch % 2 == 0:  # print loss every 100 epochs
            print("Epoch:", epoch, "Loss:", loss, "Validation F1:", f1s[-1], "Validation Acc:", accs[-1])
            
    
    return network, losses

def predict(network, X):
    
    # Get the final activations from forward propagation
    # a2 = forward_propagation(network, X)[-1]
    a4 = forward_propagation(network, X)[-1]
    
    # Convert activations to binary predictions (0 or 1)
    # predictions = (np.squeeze(a2) > 0.5).astype(int)
    predictions = (np.squeeze(a4) > 0.5).astype(int)
    
    return predictions