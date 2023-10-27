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
    # gradients=[]
    # data_size = len(y)
    # for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=data_size):
    #     # e = minibatch_y - minibatch_tx.dot(w)
    #     # gradient = -1/len(minibatch_y) * minibatch_tx.T.dot(e)  
    #     gradient = compute_gradient(minibatch_y, minibatch_tx, w)
    #     gradients.append(gradient)
    #     mse = compute_mse(minibatch_y, minibatch_tx, w)
    # avg_gradient = np.mean(gradients, axis=0)

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
        # print(
        #     "GD iter. {bi}/{ti}: loss={l}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss
        #     )
        # )
    return w, loss


""" a function used to perform regularized logistic regression using gradient descent. regularization term is L2."""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=False):
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
        if verbose:
            print(
                "GD iter. {bi}/{ti}: loss={l}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss
                )
            )
    return w, loss

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

def hinge_regression(y, tx, initial_w, max_iters, gamma, lambda_=0.1):
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
    for n_iter in range(max_iters):
        gradient = hinge_gradient(y, tx, w, lambda_)
        loss = hinge_loss(y, tx, w, lambda_)
        w = w - gamma * gradient
        # if n_iter % 50 == 0:
        #     print(
        #         "GD iter. {bi}/{ti}: loss={l}".format(
        #             bi=n_iter, ti=max_iters - 1, l=loss
        #         )
        #     )
    return w, loss



########### data preparation ###########

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

def one_hot_encoding(tx, cate_indices):
    """one-hot encoding for categorical features.
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is number of features
        cate_indices: indices of the categorical features
    Returns:
        numpy array containing the data, with categorical features one-hot encoded.
    """
    # get the indices of non-categorical features
    non_cate_indices = np.delete(np.arange(tx.shape[1]), cate_indices)
    # get the non-categorical features
    non_cate = tx[:, non_cate_indices]
    # get the categorical features
    cate = tx[:, cate_indices]
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
    x = np.copy(tx[:, std!=0])
    x = (x - mean[std!=0]) / std[std!=0]
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
# def predict_acc(x_val, y_val, best_weights, logistic=False,threshold=0.5):
#     if logistic:
#         y_pred = sigmoid(x_val @ best_weights)
#         y_pred[y_pred >= 0.5] = 1
#         y_pred[y_pred < 0.5] = 0
#     else:
#         y_pred = x_val @ best_weights
#         y_pred[y_pred >= threshold] = 1
#         y_pred[y_pred < threshold] = 0
#     accuracy = (y_pred == y_val).sum() / len(y_val)
#     print("The Accuracy is: %.4f"%accuracy)

# calculate F1 score
# def predict_f1(x_val, y_val, best_weights, logistic=False, threshold=0.5):
#     if logistic:
#         y_pred = sigmoid(x_val @ best_weights)
#         y_pred[y_pred >= 0.5] = 1
#         y_pred[y_pred < 0.5] = 0
#     else:
#         y_pred = x_val @ best_weights
#         y_pred[y_pred >= threshold] = 1
#         y_pred[y_pred < threshold] = 0
#         # print("y_pred", y_pred)
#     tp = np.sum((y_pred == 1) & (y_val == 1))
#     fp = np.sum((y_pred == 1) & (y_val == 0))
#     fn = np.sum((y_pred == 0) & (y_val == 1))
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = 2 * (precision * recall) / (precision + recall)
#     print("The F1 score is: %.4f"%f1)
#     print("The precision is: %.4f"%precision)
#     print("The recall is: %.4f"%recall)

# calculate accuracy
def predict_acc_pure(y_pred, y_val):
    accuracy = (y_pred == y_val).sum() / len(y_val)
    print("The Accuracy is: %.4f"%accuracy)
    return acc

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
    add_num = pos_num - neg_num

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
    np.random.seed(1)
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

def combinations_with_replacement(seq, r):
    """Generate combinations with replacement for seq of length r."""
    if r == 0:
        return [[]]
    if not seq:
        return []
    
    # Split the sequence to head (first element) and tail (rest of the elements)
    head, tail = seq[0], seq[1:]
    
    # First part: Combinations that do not use the head at all
    without_head = combinations_with_replacement(tail, r)
    
    # Second part: Combinations that use the head
    with_head = [[head] + rest for rest in combinations_with_replacement(seq, r-1)]
    
    return with_head + without_head

def polynomial_expansion(x, degree):
    """polynomial expansion
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        degree: degree of polynomial expansion
    Returns:
        output: numpy array containing the data, with polynomial expansion.
    """
    if degree < 1:
        raise ValueError("Degree should be at least 1.")
    
    N, D = x.shape
    output = x.copy()

    for deg in range(2, degree + 1):
        for feature_combination in combinations_with_replacement(range(D), deg):
            new_feature = np.prod(x[:, feature_combination], axis=1)
            new_feature = new_feature.reshape((N, 1))
            output = np.hstack((output, new_feature))
    
    return output

def polynomial_expansion_single(x, degree):
    """polynomial expansion
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        degree: degree of polynomial expansion
    Returns:
        output: numpy array containing the data, with polynomial expansion without combining different features.
    """
    
    N, D = x.shape
    expanded_features = [x]  # start with the original features
    
    for deg in range(2, degree + 1):
        expanded_features.append(np.power(x, deg))
        
    return np.hstack(expanded_features)



########## Neural Network ##########


# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         # Initialize weights and biases
#         self.W1 = np.random.randn(input_size, hidden_size)
#         self.b1 = np.zeros((1, hidden_size))
#         self.W2 = np.random.randn(hidden_size, output_size)
#         self.b2 = np.zeros((1, output_size))
    
#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
    
#     def sigmoid_derivative(self, z):
#         return self.sigmoid(z) * (1 - self.sigmoid(z))
    
#     def forward(self, X):
#         # Hidden layer
#         self.z1 = np.dot(X, self.W1) + self.b1
#         self.a1 = self.sigmoid(self.z1)
#         # Output layer
#         self.z2 = np.dot(self.a1, self.W2) + self.b2
#         self.a2 = self.sigmoid(self.z2)
#         return self.a2
    
#     def backward(self, X, y, learning_rate=0.01):
#         m = X.shape[0]
        
#         # Calculate the output layer error
#         dz2 = self.a2 - y
#         dW2 = 1/m * np.dot(self.a1.T, dz2)
#         db2 = 1/m * np.sum(dz2, axis=0, keepdims=True)
        
#         # Calculate the hidden layer error
#         dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
#         dW1 = 1/m * np.dot(X.T, dz1)
#         db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)
        
#         # Update the weights and biases
#         self.W2 -= learning_rate * dW2
#         self.b2 -= learning_rate * db2
#         self.W1 -= learning_rate * dW1
#         self.b1 -= learning_rate * db1
        
#     def train(self, X, y, epochs=1000, learning_rate=0.01):
#         for epoch in range(epochs):
#             # Forward pass
#             predictions = self.forward(X)
#             # Compute loss
#             loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#             # Backward pass
#             self.backward(X, y, learning_rate)
#             if epoch % 100 == 0:
#                 print(f"Epoch: {epoch}, Loss: {loss:.4f}")
#         return self

#     def predict(self, X):
#         predictions = self.forward(X)
#         return (predictions > 0.5).astype(int)

# # Example usage
# if __name__ == "__main__":
#     np.random.seed(42)
    
#     # Generate some random data for training
#     X = np.random.randn(1000, 2)
#     y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1) # Simple linear boundary

#     nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=1)
#     nn.train(X, y, epochs=1000, learning_rate=0.1)
    
#     # Test predictions
#     test_X = np.array([[0.5, 0.5], [-0.5, -0.5]])
#     predictions = nn.predict(test_X)
#     print("Test Predictions:", predictions)


import numpy as np

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
    # hidden_layer = {'W': np.random.randn(input_size, hidden_size)*0.1,
    #                 'b': np.zeros((1, hidden_size))}
    # output_layer = {'W': np.random.randn(hidden_size, output_size)*0.1,
    #                 'b': np.zeros((1, output_size))}
    # network.append(hidden_layer)
    # network.append(output_layer)

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
    # z2 = np.dot(a1, network[1]['W']) + network[1]['b']
    # a2 = sigmoid(z2)
    z2 = np.dot(a1, network[1]['W']) + network[1]['b'] 
    a2 = leaky_relu(z2)

    z3 = np.dot(a2, network[2]['W']) + network[2]['b']
    a3 = leaky_relu(z3)


    z4 = np.dot(a3, network[3]['W']) + network[3]['b']
    a4 = sigmoid(z4)

    # clip a2 values
    epsilon = 1e-7
    # a2 = np.clip(a2, epsilon, 1 - epsilon)
    # activations = [a1, a2]
    a4 = np.clip(a4, epsilon, 1 - epsilon)

    # clip z4 values
    # z4 = np.clip(z4, epsilon, 1 - epsilon)
    activations = [a1, a2, a3, a4]
    return activations

def backpropagation(network, x, y, activations):
    m = x.shape[0]
    # a1, a2 = activations
    a1, a2, a3, a4 = activations
    
    # Start from the output layer
    # dz2 = a2 - y.reshape(-1, 1)
    # dw2 = np.dot(a1.T, dz2) / m
    # db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # da1 = np.dot(dz2, network[1]['W'].T)
    # dz1 = da1 * relu_derivative(a1)
    # dw1 = np.dot(x.T, dz1) / m
    # db1 = np.sum(dz1, axis=0, keepdims=True) / m

   
    # loss = -np.mean(y * np.log(z4) + (1 - y) * np.log(1 - z4))

    dz4 = a4 - y.reshape(-1, 1)
    dw4 = np.dot(a3.T, dz4) / m
    db4 = np.sum(dz4, axis=0, keepdims=True) / m

    # dz3 = a3 - y.reshape(-1, 1)
    # dw3 = np.dot(a2.T, dz3) / m
    # db3 = np.sum(dz3, axis=0, keepdims=True) / m

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
    dw1 = np.clip(dw1, -max_int, max_int)
    dw2 = np.clip(dw2, -max_int, max_int)
    dw3 = np.clip(dw3, -max_int, max_int)
    dw4 = np.clip(dw4, -max_int, max_int)
    db1 = np.clip(db1, -max_int, max_int)
    db2 = np.clip(db2, -max_int, max_int)
    db3 = np.clip(db3, -max_int, max_int)
    db4 = np.clip(db4, -max_int, max_int)
    

    # return [{"dW": dw1, "db": db1}, {"dW": dw2, "db": db2}]
    return [{"dW": dw1, "db": db1}, {"dW": dw2, "db": db2}, {"dW": dw3, "db": db3}, {"dW": dw4, "db": db4}]

def update_weights(network, gradients, learning_rate):
    for layer, gradient in zip(network, gradients):
        layer['W'] -= learning_rate * gradient['dW']
        layer['b'] -= learning_rate * gradient['db']

def train(network, X, y, learning_rate, epochs, batch_size):
    m = X.shape[0]
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
            # loss = -np.mean(y_batch * np.log(a4) + (1 - y_batch) * np.log(1 - a4))
            logprobs = np.dot(y_batch.reshape(1,-1),np.log(a4))+np.dot((1-y_batch.reshape(1,-1)),np.log(1-a4)) 
            loss = -logprobs / m
        
            gradients = backpropagation(network, X_batch, y_batch, activations)
            update_weights(network, gradients, learning_rate)
        
        if epoch % 2 == 0:  # print loss every 100 epochs
            print("Epoch:", epoch, "Loss:", loss)
    
    return network

def predict(network, X):
    
    # Get the final activations from forward propagation
    # a2 = forward_propagation(network, X)[-1]
    a4 = forward_propagation(network, X)[-1]
    
    # Convert activations to binary predictions (0 or 1)
    # predictions = (np.squeeze(a2) > 0.5).astype(int)
    predictions = (np.squeeze(a4) > 0.5).astype(int)
    
    return predictions



##### Nerual Network with keras  #####
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils

# # for modeling
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStoppingpip

