import numpy as np

"""a function used to compute the mean squared error."""

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    MSE = np.sum((y - tx.dot(w))**2) / (2 * len(y))
    return MSE


""" a function used to compute mean squared error gradient."""

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    e = y - tx.dot(w)
    
    gradient = -1/N * tx.T.dot(e)
    return gradient


""" a function used to perform mean squared error gradient descent."""

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of the method
        loss: the last loss value
        
    """
    # initialize w
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
        print(
            "GD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )

    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


""" a function used to compute stochastic gradient."""
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    gradients=[]
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
        e = minibatch_y - minibatch_tx.dot(w)
        gradient = -1/len(minibatch_y) * minibatch_tx.T.dot(e)
        gradients.append(gradient)
        mse = compute_mse(minibatch_y, minibatch_tx, w)
    avg_gradient = np.mean(gradients, axis=0)
    return avg_gradient,mse
 

""" a function used to perform mse stochastic gradient descent."""
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: mean squared error of the last minibatch of the last SGD iteration
        w: the last weight vector of the method
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient,loss = compute_stoch_gradient(y, tx, w)
        w = w - gamma * gradient

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, loss


""" a function used to perform least squares regression using normal equations."""
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse scalar.
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse(y, tx, w)
    return w, loss


""" a function used to perform ridge regression using normal equations."""
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse scalar.
    """
    N,D = tx.shape
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(D), tx.T.dot(y))
    loss = compute_mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1.0 / (1 + np.exp(-t))


""" a function used to calculate logistic regression loss by negative log likelihood."""
def log_likely_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    pred_probs = sigmoid(tx.dot(w))
    total_loss = -np.sum(y * np.log(pred_probs) + (1 - y) * np.log(1 - pred_probs))
    
    # Calculate mean loss
    loss = total_loss / len(y)
    return loss

""" a function used to calculate negative log likelihood gradient."""
def log_likely_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        gradient of the loss with respect to w, shape=(D, 1)
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    pred_probs = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred_probs - y) / len(y)
    return gradient

""" a function used to perform logistic regression using gradient descent."""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: negative log likelyhood cost.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = log_likely_gradient(y, tx, w)
        loss = log_likely_loss(y, tx, w)
        w = w - gamma * gradient
        print(
            "GD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    return w, loss


""" a function used to perform regularized logistic regression using gradient descent. regularization term is L2."""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: negative log likelyhood cost.
    """
    w = initial_w
    loss = log_likely_loss(y, tx, w)

    for n_iter in range(max_iters):
        gradient = log_likely_gradient(y, tx, w) + 2 * lambda_ * w
        loss = log_likely_loss(y, tx, w) 
        w = w - gamma * gradient
        print(
            "GD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    return w, loss


########### data preparation ###########
def add_bias(x):
    """add bias to the data.

    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features

    Returns:
        numpy array containing the data, with bias added.
    """
    return np.hstack((np.ones((x.shape[0], 1)), x))

def fillna_with_mean(tx, threshold=0.5):
    """
    replace the missing value with mean value of each feature, and remove features where over 50% of the data is NaN.     
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        numpy array containing the data, with missing values replaced with mean.
    """
    # remove columns where over 50% of the data is NaN
    nan_percentages = np.sum(np.isnan(tx), axis=0) / tx.shape[0]
    x = np.copy(tx)

    x = x[:, nan_percentages < threshold]

    for feature in range(x.shape[1]):
        nan_mask = np.isnan(x[:,feature])
        clean_data = x[~nan_mask,feature]
        x[nan_mask, feature] = np.mean(clean_data)
    return x

def standardize(tx):
    """z-score standardization
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        output: standardized data, mean at 1 and std at 1

    """
    # for training set:
    mean = tx.mean(axis=0)
    std = tx.std(axis=0)
    std[std == 0] = 1
    x = np.copy(tx)
    x = (x - mean) / std
    return x

def process_y(ty):
    """converts the labels from {-1,1} to {0,1}
    Args:
        ty: numpy array of shape (N,), N is the number of samples
    Returns:
        numpy array containing the labels, with {-1,1} converted to {0,1}
    """
    y = np.copy(ty)
    y[y == -1] = 0
    return y

# calculate accuracy
def predict_acc(x_val, y_val, best_weights, logistic=False,threshold=0.5):
    if logistic:
        y_pred = sigmoid(x_val @ best_weights)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
    else:
        y_pred = x_val @ best_weights
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
    accuracy = (y_pred == y_val).sum() / len(y_val)
    print("The Accuracy is: %.4f"%accuracy)

# calculate F1 score
def predict_f1(x_val, y_val, best_weights, logistic=False, threshold=0.5):
    if logistic:
        y_pred = sigmoid(x_val @ best_weights)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
    else:
        y_pred = x_val @ best_weights
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        # print("y_pred", y_pred)
    tp = np.sum((y_pred == 1) & (y_val == 1))
    fp = np.sum((y_pred == 1) & (y_val == 0))
    fn = np.sum((y_pred == 0) & (y_val == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("The F1 score is: %.4f"%f1)
    print("The precision is: %.4f"%precision)
    print("The recall is: %.4f"%recall)

# feature selection using PCA
def pca(x, num_components):
    """PCA algorithm
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        num_components: number of components to keep
    Returns:
        output: numpy array containing the data, with missing values replaced with mean.
    """
    # standardize the data
    x = standardize(x)
    # calculate covariance matrix
    cov = np.cov(x.T)
    # calculate eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eig(cov)
    weight = eig_val / np.sum(eig_val)
    # sort eigenvalues and eigenvectors
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]
    weight = weight[idx]
    cummulation = np.cumsum(weight)
    # select the first num_components eigenvectors
    eig_vec = eig_vec[:, :num_components]
    # project the data onto the new basis
    x = x @ eig_vec
    return x, eig_vec, eig_val, weight
