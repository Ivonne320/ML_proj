import numpy as np

# calculate accuracy
def predict_acc_pure(y_pred, y_val):
    accuracy = (y_pred == y_val).sum() / len(y_val)
    # print("The Accuracy is: %.4f"%accuracy)
    return accuracy

# calculate F1 score
def predict_f1_pure(y_pred, y_val):

    tp = np.sum((y_pred == 1) & (y_val == 1))
    fp = np.sum((y_pred == 1) & (y_val != 1))
    fn = np.sum((y_pred != 1) & (y_val == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    # print("The F1 score is: %.4f"%f1)
    # print("The precision is: %.4f"%precision)
    # print("The recall is: %.4f"%recall)
    return f1

class NeuralNetwork:
    
    def __init__(self, layer_sizes, output_activation='sigmoid', loss_function='bce', adam=False):
        # used for adam optimizer
        self.adam = adam
        self.network = self.initialize_network(layer_sizes)
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.seed = 20
        np.random.seed(self.seed)

        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def initialize_network(self, sizes):
        network = []
        if self.adam:
            self.mom = []
            self.v = []
        for i in range(len(sizes) - 1):
            layer = {
                'W': np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2./sizes[i]),
                'b': np.zeros((1, sizes[i+1]))
            }
            network.append(layer)
            if self.adam:
                self.mom.append(np.zeros((sizes[i], sizes[i+1])))
                self.v.append(np.zeros((sizes[i], sizes[i+1])))
        return network

    def forward_propagation(self, x):
        activations = []
        a = x
        for i, layer in enumerate(self.network):
            z = np.dot(a, layer['W']) + layer['b']
            if i != len(self.network) - 1:  # if not output layer
                a = self.relu(z)
            else:  # output layer
                if self.output_activation == 'sigmoid':
                    a = self.sigmoid(z)
                elif self.output_activation == 'relu':
                    a = self.relu(z)
                elif self.output_activation == 'linear':
                    a = z
                else:
                    raise ValueError("Invalid output activation function.")       
            activations.append(a)
        return activations
    
    def compute_loss(self, y_pred, y_true):
        if self.loss_function == 'bce':
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_function == 'mse':
            return np.mean((y_true - y_pred) ** 2, dtype=np.float128)
        else:
            raise ValueError("Invalid loss function.")

    def backpropagation(self, x, y, activations):
        m = x.shape[0]
        gradients = []
        # da = activations[-1] - y.reshape(-1, 1)
        # depending on the loss function, the derivative of the last layer
        if self.loss_function == 'bce':
            da = activations[-1] - y.reshape(activations[-1].shape)
        elif self.loss_function == 'mse':
            da = 2 * (activations[-1] - y.reshape(activations[-1].shape))
        else:
            raise ValueError("Invalid loss function.")

        for i in reversed(range(len(self.network))):
            if i == len(self.network) - 1:
                if self.output_activation == 'sigmoid':
                    dz = da * activations[i] * (1 - activations[i])
                elif self.output_activation == 'relu':
                    dz = da * self.relu_derivative(activations[i])
                elif self.output_activation == 'linear':
                    dz = da
                else:
                    raise ValueError("Invalid output activation function.")
            else: # hidden layers always use relu
                dz = da * self.relu_derivative(activations[i])
            dw = np.dot(activations[i - 1].T if i != 0 else x.T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            gradients.insert(0, {"dW": dw, "db": db})
            da = np.dot(dz, self.network[i]['W'].T)
        return gradients

    def update_weights(self, gradients, learning_rate):
        for layer, gradient in zip(self.network, gradients):
            layer['W'] -= learning_rate * gradient['dW']
            layer['b'] -= learning_rate * gradient['db']

    def update_weights_adam(self, gradients, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """update weights using adam optimizer"""
        for n, (layer, gradient) in enumerate(zip(self.network, gradients)):
            self.mom[n] = beta1*self.mom[n] + (1-beta1)*gradient['dW']
            self.v[n] = beta2*self.v[n] + (1-beta2)*(gradient['dW']**2)   
            m_hat = self.mom[n]/(1-beta1**t)
            v_hat = self.v[n]/(1-beta2**t)
            layer['W'] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            layer['b'] -= learning_rate * gradient['db']


    def train(self, X, y, X_v=None, y_v=None, learning_rate=0.01, epochs=100, batch_size=2048, n_pat=10, early_stop=True):
        #... (similar to original implementation, using class methods)
        m = X.shape[0]
        losses = []
        v_losses = []
        f1s = []
        accs = []
        best_f1 = -np.inf
        best_loss = np.inf
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
                
                activations = self.forward_propagation(X_batch)
                # a2 = activations[-1]
                # loss = -np.mean(y_batch * np.log(a2) + (1 - y_batch) * np.log(1 - a2))
                a4 = activations[-1]
                loss = self.compute_loss(a4, y_batch.reshape(a4.shape))
                losses.append(loss)
                # logprobs = np.dot(y_batch.reshape(1,-1),np.log(a4))+np.dot((1-y_batch.reshape(1,-1)),np.log(1-a4)) 
                # loss = -logprobs / m
            
                gradients = self.backpropagation(X_batch, y_batch, activations)
                if self.adam:
                    self.update_weights_adam(gradients, learning_rate, epoch+1)
                else:
                    self.update_weights(gradients, learning_rate)
                    
            
            if early_stop and X_v is not None and y_v is not None:
                # test the model on validation set
                if self.loss_function == 'bce':
                    threshold = np.arange(0.1, 1, 0.1)
                    y_pred = [(self.forward_propagation(X_v)[-1].squeeze() > thres).astype(int) for thres in threshold]
                    f1s.append([predict_f1_pure(y_pred[i], y_v) for i in range(len(threshold))])
                    accs.append([predict_acc_pure(y_pred[i], y_v) for i in range(len(threshold))])

                    # early stopping
                    if np.max(f1s[-1]) > best_f1:
                        best_f1 = np.max(f1s[-1])
                        best_threshold = threshold[np.argmax(f1s[-1])]
                        self.network[-1]["best_threshold"] = best_threshold
                        self.network[-1]["best_f1"] = best_f1
                        num_epochs_without_improvement = 0
                        best_network = self.network.copy()
                    else:
                        num_epochs_without_improvement += 1
                        if num_epochs_without_improvement > patience:
                            print("Early stopping at epoch", epoch)
                            self.network = best_network
                            break
                    # print index every 2 epochs
                    if epoch % 2 == 0:  # print loss every 100 epochs
                        print("Epoch:", epoch, "Loss:", loss, "Validation F1:", f1s[-1], "Validation Acc:", accs[-1])

                elif self.loss_function == 'mse':
                    output_v = self.get_output(X_v)
                    loss_v = self.compute_loss(output_v, y_v.reshape(output_v.shape))
                    v_losses.append(loss_v)
                    if loss_v < best_loss:
                        best_loss = loss_v
                        num_epochs_without_improvement = 0
                        best_network = self.network.copy()
                    else:
                        num_epochs_without_improvement += 1
                        if num_epochs_without_improvement > patience:
                            print("Early stopping at epoch", epoch)
                            self.network = best_network
                            break
                    if epoch % 2 == 0:  # print loss every 100 epochs
                        print("Epoch:", epoch, "Loss:", loss, "Validation Loss:", loss_v)

            else:
                if epoch % 2 == 0:  # print loss every 100 epochs
                    print("Epoch:", epoch, "Loss:", loss)
            
        return self.network, losses, v_losses
        
    def predict(self, X):
        return (np.squeeze(self.forward_propagation(X)[-1]) > 0.5).astype(int)
    
    def get_feature(self, X, num_layer):
        """return the feature using autoencoder"""
        return self.forward_propagation(X)[num_layer]
    
    def get_output(self, X):
        """return the output of the network"""
        return self.forward_propagation(X)[-1]
