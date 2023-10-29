import numpy as np

################################
# Regression methods
################################
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
    
    gradient = -1/N * tx.T@e
    return gradient


""" a function used to perform mean squared error gradient descent."""

def mean_square_error_gd(y, tx, initial_w, max_iters, gamma):
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
        if n_iter % 10 == 0:
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
        if start_index != end_index: # maybe <= end_index?
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

    data_size = len(y)
    e = y - tx.dot(w)

    gradient = -1/data_size * (tx.T @ e)

    return gradient
 

""" a function used to perform mse stochastic gradient descent."""
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
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
        loss = compute_mse(y, tx, w)
        for batch_y, batch_tx in batch_iter(y, tx, batch_size):
            stoch_grad = compute_stoch_gradient(batch_y, batch_tx, w)
            w = w - gamma * stoch_grad
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return loss, w


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

""" a function used to perform ridge regression using normal equations."""
def ridge_regression_var(y, tx, y_v, tx_v, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse scalar.
    """

    f1s = []
    accs = []
    best_f1 = -np.inf
    best_threshold = 0

    N,D = tx.shape
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(D), tx.T.dot(y))
    loss = compute_mse(y, tx, w)

    threshold = np.arange(0.1, 1, 0.1)
    y_pred = [(tx_v @ w > thres).astype(int) for thres in threshold]
    f1s = [predict_f1_pure(y_pred[i], y_v) for i in range(len(threshold))]
    accs = [predict_acc_pure(y_pred[i], y_v) for i in range(len(threshold))]
    best_f1 = np.max(f1s[-1])
    best_threshold = threshold[np.argmax(f1s[-1])]

    return w, loss, best_f1, best_threshold


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
def logistic_regression(y, tx, y_v, tx_v, initial_w, max_iters, gamma):
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
        # print(
        #     "GD iter. {bi}/{ti}: loss={l}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss
        #     )
        # )
    return w, loss


""" a function used to perform regularized logistic regression using gradient descent. regularization term is L2."""
def reg_logistic_regression(y, tx, y_v, tx_v, lambda_, initial_w, max_iters, gamma, n_pat=20):
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
    for n_iter in range(max_iters):
        gradient = log_likely_gradient(y, tx, w) + 2 * lambda_ * w
        loss = log_likely_loss(y, tx, w)
        w = w - gamma * gradient

        print("Loss at iteration ", n_iter, ":", loss)

    return w, loss

def reg_logistic_regression_var(y, tx, y_v, tx_v, lambda_, initial_w, max_iters, gamma, n_pat=20):
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
    best_weight = initial_w
    losses = []
    f1s = []
    accs = []
    best_f1 = -np.inf
    best_threshold = 0
    num_epochs_without_improvement = 0
    patience = n_pat
    for n_iter in range(max_iters):
        gradient = log_likely_gradient(y, tx, w) + 2 * lambda_ * w
        loss = log_likely_loss(y, tx, w)
        w = w - gamma * gradient

        losses.append(loss)
        # test the model on validation set
        threshold = np.arange(0.1, 1, 0.1)
        y_pred = [(sigmoid(tx_v @ w) > thres).astype(int) for thres in threshold]
        f1s.append([predict_f1_pure(y_pred[i], y_v) for i in range(len(threshold))])
        accs.append([predict_acc_pure(y_pred[i], y_v) for i in range(len(threshold))])

        # early stopping
        if np.max(f1s[-1]) > best_f1:
            best_f1 = np.max(f1s[-1])
            best_threshold = threshold[np.argmax(f1s[-1])]
            num_epochs_without_improvement = 0
            best_weight = w.copy()
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement > patience:
                print("Early stopping at iteration ", n_iter)
                w = best_weight
                break

    return w, losses, best_f1, best_threshold

def hinge_loss(y, tx, w, lambda_=0.1):
    """Compute the hinge loss.
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape (D,), D is the number of features.
        lambda_: scalar.
    Returns:
        hinge loss
    """
    N = len(y)
    pred = tx.dot(w)
    loss = np.sum(np.maximum(0, 1 - y * pred)) / N + lambda_ / 2 * np.sum(w**2)

    return loss

def hinge_gradient(y, tx, w, lambda_=0.1):
    """Compute the subgradient of hinge loss.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape (D,), D is the number of features.
        lambda_: scalar.
    Returns:
        subgradient of hinge loss
    """
    N = len(y)
    pred = tx.dot(w)
    mask = np.where(y * pred < 1, 1, 0)
    gradient = -1/N * tx.T.dot(y * mask) + 2 * lambda_ * w

    return gradient

def hinge_predict(tx, w):
    """predict the labels using hinge regression.
    Args:
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape (D,), D is the number of features.
    Returns:
        predicted labels
    """
    pred = tx.dot(w)
    pred[pred > 0] = 1
    pred[pred <= 0] = 0
    return pred

def hinge_regression(y, tx, y_v, tx_v, initial_w, max_iters, gamma, lambda_=0.1, n_pat=20):
    """implement hinge regression using subgradient descent.
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), D is the number of features.
        max_iters: scalar.
        gamma: scalar.
        lambda_: scalar.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: hinge loss."""
    w = initial_w
    best_weight = initial_w
    best_threshold = 0
    losses = []
    f1s = []
    accs = []
    best_f1 = -np.inf
    num_epochs_without_improvement = 0
    patience = n_pat
    y[y==0] = -1
    y_v[y_v==0] = -1
    for n_iter in range(max_iters):
        gradient = hinge_gradient(y, tx, w, lambda_)
        loss = hinge_loss(y, tx, w, lambda_)
        w = w - gamma * gradient

        losses.append(loss)
        # test the model on validation set
        threshold = np.arange(0.2, 2, 0.2)
        y_pred = [(tx_v @ w > thres).astype(int) for thres in threshold]
        f1s.append([predict_f1_pure(y_pred[i], y_v) for i in range(len(threshold))])
        accs.append([predict_acc_pure(y_pred[i], y_v) for i in range(len(threshold))])
        # early stopping
        if np.max(f1s[-1]) > best_f1:
            best_f1 = np.max(f1s[-1])
            best_threshold = threshold[np.argmax(f1s[-1])]
            num_epochs_without_improvement = 0
            best_weight = w.copy()
        else:
            num_epochs_without_improvement += 1
            if num_epochs_without_improvement > patience:
                print("Early stopping at iteration ", n_iter)
                w = best_weight
                break

    return w, losses, best_f1, best_threshold


################################
# Data preprocessing methods
################################
def normalize_nan(x):
    """Uniform encoding for missing values as NaN. Feature values over 95 percentile of the column (distribution without 
    existing "NaN") are considered as NaN.
    
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
    
    Returns:
        numpy array containing the data, with missing values encoded as NaN.
    
    """ 
    x = np.copy(x)
    for feature in range(x.shape[1]):
        # get the 95 percentile of the feature
        percentile = np.percentile(x[:,feature], 95)
        # replace values over 95 percentile with NaN
        x[x[:,feature] > percentile, feature] = np.nan
    return x

def drop_rows(x, y, threshold = 0.5):
    """drop rows where over threshold of the data is NaN.
    
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        y: numpy array of shape (N,), N is the number of samples
        threshold: threshold for dropping rows
    
    Returns:
        numpy array containing the data, with rows dropped.
    """
    # get the number of NaN in each row
    nan_num = np.sum(np.isnan(x), axis=1)
    # get the indices of rows with less than threshold NaN
    indices = np.where(nan_num < threshold * x.shape[1])[0]
    return x[indices], y[indices], indices



def add_bias(x):
    """add bias to the data.

    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features

    Returns:
        numpy array containing the data, with bias added.
    """
    return np.hstack((np.ones((x.shape[0], 1)), x))

def drop_features(x, threshold=0.5):
    """drop features where over threshold of the data is NaN and also those with std == 0.
    
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        threshold: threshold for dropping features

    Returns:
        x: numpy array containing the data, with features dropped.
        indices: indices of the features kept.
    """
    # get the number of NaN in each feature
    nan_num = np.sum(np.isnan(x), axis=0)
    # get the indices of features with less than threshold NaN and std != 0, calculate the std for each feature excluding NaN
    # indices = np.where(nan_num < threshold * x.shape[0])

    indices = np.where((nan_num < threshold * x.shape[0]) & (np.nanstd(x, axis=0) != 0))[0]

    return x[:, indices], indices

def check_categorical(x, threshold = 10):
    """check if there are categorical features in the data.
    
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features

    Returns:
        indices: indices of the categorical features.
    """
    indices = []
    for feature in range(x.shape[1]):
        if len(np.unique(x[:, feature])) < threshold:
            indices.append(feature)
    return indices

def fillna(tx, cate_indices):
    """
    replace the missing value with mean value for non-categorical features and majority label for categorical features .     
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        numpy array containing the data, with missing values replaced.
    """
    
    x = np.copy(tx)
    for feature in range(x.shape[1]):
        if feature not in cate_indices:
            # replace NaN with mean
            mean = np.nanmean(x[:, feature])
            x[np.isnan(x[:, feature]), feature] = mean
        else:
            # replace NaN with majority label if the label is not NaN, otherwise replace the second majority label
            unique, counts = np.unique(x[:, feature], return_counts=True)
            majority = unique[np.argmax(counts)]
            if np.isnan(majority):
                majority = unique[np.argsort(counts)[-2]]
            else:
                majority = unique[np.argsort(counts)[-1]]
            x[np.isnan(x[:, feature]), feature] = majority
    return x

def one_hot_encoding(tx, test_x, cate_indices):
    """one-hot encoding for categorical features.
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is number of features
        cate_indices: indices of the categorical features
    Returns:
        numpy array containing the data, with categorical features one-hot encoded.
    """
    # get the indices of non-categorical features
    split = tx.shape[0]
    new_x = np.vstack((tx, test_x))
    non_cate_indices = np.delete(np.arange(new_x.shape[1]), cate_indices)
    # get the non-categorical features
    non_cate = new_x[:, non_cate_indices]
    # get the categorical features
    cate = new_x[:, cate_indices]
    # one-hot encoding
    new_cate = []
    for feature in range(cate.shape[1]):
        for unique in np.unique(cate[:, feature]):
            new_cate.append((cate[:, feature] == unique).astype(int))
    # cate = np.array([np.eye(len(np.unique(cate[:, feature])))[cate[:, feature].astype(int)] for feature in range(cate.shape[1])])
    # cate = np.transpose(cate, (1,2,0))
    # concatenate the features
    new_cate = np.array(new_cate).T
    x = np.hstack((non_cate, new_cate))
    return x[:split], x[split:]

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
    std[std==0] = 1
    x = np.copy(tx)
    x = (x - mean) / std
    return x, mean, std

def normalization(tx, max_=None, min_=None):
    """normalization
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        output: normalized data, range from 0 to 1

    """
    if max_ is None or min_ is None:
        # for training set:
        max_ = tx.max(axis=0)
        min_ = tx.min(axis=0)
    x = np.copy(tx)
    indices = np.where(max_ == min_)[0]
    max_[indices] = 1
    min_[indices] = 0
    x = (x - min_) / (max_ - min_)
    return x, max_, min_

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

def data_augmentation(x, y):
    """ Increase datapoints to balance the dataset
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        y: numpy array of shape (N,), N is the number of samples
        
    Returns:
        new_x: numpy array of shape (N,D), N is the number of samples, D is number of features
        new_y: numpy array of shape (N,), N is the number of samples
    """

    # get the indices of the positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y != 1)[0]

    # shuffle the indices
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # get the number of positive and negative samples
    pos_num = len(pos_indices)
    neg_num = len(neg_indices)

    # get the number of samples to add
    add_num = (pos_num - neg_num) * 2

    # get the indices of the samples to add
    if add_num > 0:
        add_indices = np.random.choice(neg_indices, add_num)
    else:
        add_num = -add_num
        add_indices = np.random.choice(pos_indices, add_num)

    # add the samples
    new_x = np.vstack((x, x[add_indices]))
    new_y = np.hstack((y, y[add_indices]))

    return new_x, new_y

def z_outlier_removal(standardized_x, y, z_threshold=2.5, feature_threshold=0.3):
    """remove outliers using z-score, regard a datapoint having more than 30% of the features 
    with Z-score>2.5 as outliers, remove from x and corresponding y
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        y: numpy array of shape (N,), N is the number of samples
        threshold: threshold for outlier removal
    Returns:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        y: numpy array of shape (N,), N is the number of samples
        
    """
    # calculate the z-score
    z = np.abs(standardized_x)
    # get the number of features
    feature_num = z.shape[1]
    # get the number of features with z-score > threshold
    feature_count = np.sum(z > z_threshold, axis=1)
    # get the indices of the samples to remove
    indices = np.where(feature_count > feature_threshold * feature_num)[0]
    # remove the samples
    x = np.delete(standardized_x, indices, axis=0)
    y = np.delete(y, indices, axis=0)
    return x, y

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
    x, _, _ = standardize(x)
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

################################
# Data split tools
################################

def split_data(x, y, scale):
    """split the data into training set and validation set
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        y: numpy array of shape (N,), N is the number of samples
        scale: ratio of training set
    Returns:
        x_train: numpy array of shape (N,D), N is the number of samples, D is number of features
        y_train: numpy array of shape (N,), N is the number of samples
        x_val: numpy array of shape (N,D), N is the number of samples, D is number of features
        y_val: numpy array of shape (N,), N is the number of samples
    """
    # shuffle the data
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # split the data
    split = int(len(y) * scale)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]

    return x_train, y_train, x_val, y_val

def split_cross_validation(x, y, slots = 10):
    """split the data into #_slots of sub-sets for cross validation
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        y: numpy array of shape (N,), N is the number of samples
        slots: number of sub-sets
    Returns:
        sub_x: numpy array of shape (slots,N,D), N is the number of samples, D is number of features
        sub_y: numpy array of shape (slots,N), N is the number of samples
        
    """
    # shuffle the data
    indices = np.arange(len(y))
    # np.random.seed(1)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # split the data
    split = int(len(y) / slots)
    sub_x = []
    sub_y = []
    for i in range(slots-1):
        sub_x.append(x[i*split:(i+1)*split])
        sub_y.append(y[i*split:(i+1)*split])
    sub_x.append(x[(slots-1)*split:])
    sub_y.append(y[(slots-1)*split:])
    return sub_x, sub_y


################################
# Evaluation methods
################################
def predict_acc_pure(y_pred, y_val):
    accuracy = (y_pred == y_val).sum() / len(y_val)
    # print("The Accuracy is: %.4f"%accuracy)
    return np.clip(accuracy, 0, 1)

# calculate F1 score
def predict_f1_pure(y_pred, y_val):

    tp = np.sum((y_pred == 1) & (y_val == 1))
    fp = np.sum((y_pred == 1) & (y_val != 1))
    fn = np.sum((y_pred != 1) & (y_val == 1))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    # print("The F1 score is: %.4f"%f1)
    # print("The precision is: %.4f"%precision)
    # print("The recall is: %.4f"%recall)
    return np.clip(f1, 0, 1)


################################
# Hyperparameter tuning
################################
# def random_search(x, y, model, params, slots=10, n_iter=100):


################################
# Nerual network
################################
class NeuralNetwork:

    """A neural network with adjustable size of hidden layers and activation functions.
        activtaion functions: relu (valid for hidden layers), sigmoid, linear (valid for output layer)).
        loss functions: bce (binary cross entropy), mse (mean squared error).
        weight update methods: gradient descent, adam."""
    
    def __init__(self, layer_sizes, output_activation='sigmoid', loss_function='bce', adam=False):
        # used for adam optimizer
        self.adam = adam
        self.network = self.initialize_network(layer_sizes)
        self.output_activation = output_activation
        self.loss_function = loss_function
        # self.seed = 20
        # np.random.seed(self.seed)

        
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
        """initialize the weights and biases for the given network size; 
        initialize the momentum and velocity for adam optimizer if needed
        Returns:
            network: a list of dictionaries containing the weights and biases for each layer"""
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
        """forward propagation
        Args:
            x: numpy array of shape (N,D), N is the number of samples, D is number of features
        Returns:
            activations: a list of numpy arrays containing the activations for each layer"""
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
        """compute the loss based on the given loss function"""
        if self.loss_function == 'bce':
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_function == 'mse':
            return np.mean((y_true - y_pred) ** 2, dtype=np.float128)
        else:
            raise ValueError("Invalid loss function.")

    def backpropagation(self, x, y, activations):
        """backpropagation
        Args:
            x: numpy array of shape (N,D), N is the batch size, D is number of features
            y: numpy array of shape (N,)
            activations: a list of numpy arrays containing the activations for each layer
        Returns:
            gradients: a list of dictionaries containing the gradients for each layer"""
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
        """update weights using gradient descent"""
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
        """train the model; combine with validation set and early stopping if needed;
            for validation, based on the objective, we use different metrics:
            for binary cross entropy, we use F1 score for validation set;
            for mean squared error, we use validation loss.
        Returns:
            network: a list of dictionaries containing the weights and biases for each layer
            losses: a list of losses for each epoch
            v_losses: a list of validation losses for each epoch"""
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
        """predict the labels using the output of the network"""
        return (np.squeeze(self.forward_propagation(X)[-1]) > 0.5).astype(int)
    
    def get_feature(self, X, num_layer):
        """return the feature using autoencoder"""
        return self.forward_propagation(X)[num_layer]
    
    def get_output(self, X):
        """return the output of the network"""
        return self.forward_propagation(X)[-1]