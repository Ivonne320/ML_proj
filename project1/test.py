import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import *
import optuna
from optuna.samplers import TPESampler
from optuna.samplers import CmaEsSampler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from networks import *
import datetime

PCA = True
HINGE = True
np.random.seed(10)
# x_train, x_test, y_train, train_ids, test_ids = load_csv_data_new("/home/zewzhang/Course/ML/ML_course/projects/project1/data/dataset_to_release", sub_sample=True, num=20000)
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/home/zewzhang/Course/ML/ML_course/projects/project1/data/dataset_to_release", sub_sample=True)


def data_processing(x_train, y_train, row_nan, feature_nan, z_threshold, feature_threshold, x_test):

    x_train_processed = x_train.copy()
    y_train_processed = y_train.copy()
    # transform y to 0-1 encoding
    y_train_processed = process_y(y_train_processed)

    # Uniform missing value encoding
    # x_train_processed = normalize_nan(x_train_processed)
    # Remove rows with too many nans
    x_train_processed, y_train_processed, row_indices = drop_rows(x_train_processed, y_train_processed, row_nan) # 0.55 remains 6101 rows
    # Remove features with too many nans
    x_train_processed, nan_indices = drop_features(x_train_processed, feature_nan) # 0.5 remains 174 features
    x_test_processed = x_test[:, nan_indices].copy()
    threshold_cat = 10
    # get categorical feature indices
    cat_indices = check_categorical(x_train_processed, threshold_cat)
    # handling remaining nans
    x_train_processed = fillna(x_train_processed, cat_indices)
    x_test_processed = fillna(x_test_processed, cat_indices)
    # One hot encoding for categorical features
    x_train_processed, x_test_processed = one_hot_encoding(x_train_processed, x_test_processed, cat_indices)
    x_train_processed, train_mean, train_std = standardize(x_train_processed)
    x_test_processed = (x_test_processed - train_mean) / train_std
    x_train_processed, y_train_processed = z_outlier_removal(x_train_processed, y_train_processed, z_threshold, feature_threshold)

    return x_train_processed, y_train_processed, x_test_processed

def objective_hinge(trial):
    global x_train, x_test, y_train, train_ids, test_ids
    lambda_ = trial.suggest_float('lambda_', 5e-3, 5)
    n_com = trial.suggest_int('n_com', 10, 200)
    gamma = trial.suggest_float('gamma', 0.001, 0.2)
    row_nan = trial.suggest_float('row_nan', 0.5, 0.7)
    feature_nan = trial.suggest_float('feature_nan', 0.2, 0.5)
    z_threshold = trial.suggest_float('z_threshold', 1.8, 2.5)
    feature_threshold = trial.suggest_float('feature_threshold', 0.1, 0.3)

    pre_train_data, y_train_processed, _ = data_processing(x_train, y_train, row_nan, feature_nan, z_threshold, feature_threshold, x_test)
    if PCA:
        x_pca, eig_vec, eig_val,weight = pca(pre_train_data, n_com)
        x_pca = np.real(x_pca)
        x_pca_t = x_pca.copy()
        sub_x, sub_y = split_cross_validation(x_pca_t, y_train_processed, 5)
    else:
        sub_x, sub_y = split_cross_validation(pre_train_data, y_train_processed, 5)
    f1s = []
    losss = []
    # cross-validation
    for i in range(5):
        sub_cur_x = sub_x.copy()
        sub_cur_y = sub_y.copy()
        x_v, y_v = sub_cur_x.pop(i), sub_cur_y.pop(i)
        x_t, y_t = np.vstack(sub_cur_x), np.hstack(sub_cur_y)
        x_t, y_t = data_augmentation(x_t, y_t)
        initial_w = np.random.randn(x_t.shape[1]) * 0.01
        w, loss, best_f1, best_threshold = hinge_regression(y_t, x_t, y_v, x_v, initial_w, lambda_=lambda_, max_iters=500, gamma=gamma)

        f1s.append(best_f1)
        losss.append(loss[-1])

    print("Average f1 score is: ", np.mean(f1s))
    print("Average loss is: ", np.mean(losss))

    return np.mean(f1s)

def objective_log(trial):
    global x_train, x_test, y_train, train_ids, test_ids
    lambda_ = trial.suggest_float('lambda_', 5e-3, 2)
    n_com = trial.suggest_int('n_com', 10, 200)
    gamma = trial.suggest_float('gamma', 0.001, 0.2)
    # row_nan = trial.suggest_float('row_nan', 0.5, 0.7)
    row_nan = 0.5
    # feature_nan = trial.suggest_float('feature_nan', 0.5, 0.7)
    feature_nan = 0.2
    z_threshold = trial.suggest_float('z_threshold', 1.8, 2.5)
    feature_threshold = trial.suggest_float('feature_threshold', 0.1, 0.3)
    pre_train_data, y_train_processed, x_test_processed = data_processing(x_train, y_train, row_nan, feature_nan, z_threshold, feature_threshold, x_test)
    if PCA:
        x_pca, eig_vec, eig_val,weight = pca(pre_train_data, n_com)
        x_pca = np.real(x_pca)
        x_pca_t = add_bias(x_pca)
        sub_x, sub_y = split_cross_validation(x_pca_t, y_train_processed, 5)
    else:
        pre_train_data = add_bias(pre_train_data)
        sub_x, sub_y = split_cross_validation(pre_train_data, y_train_processed, 5)
    accs = []
    f1s = []
    losss = []
    # cross-validation
    for i in range(5):
        sub_cur_x = sub_x.copy()
        sub_cur_y = sub_y.copy()
        x_v, y_v = sub_cur_x.pop(i), sub_cur_y.pop(i)
        x_t, y_t = np.vstack(sub_cur_x), np.hstack(sub_cur_y)
        x_t, y_t = data_augmentation(x_t, y_t)
        initial_w = np.random.randn(x_t.shape[1]) * 0.01
        # w, loss = logistic_regression(y_t, x_t, initial_w, max_iters=200, gamma=gamma)
        w, loss, best_f1, best_threshold = reg_logistic_regression(y_t, x_t, y_v, x_v, lambda_=lambda_, initial_w=initial_w, max_iters=500, gamma=gamma)

        f1s.append(best_f1)
        losss.append(loss[-1])

    print("Average f1 score is: ", np.mean(f1s))
    print("Average loss is: ", loss[-1])

    return np.mean(f1s)

def create_model(study):
    model = {}
    x_train1, x_test1, y_train1, _, _ = load_csv_data_new("/home/zewzhang/Course/ML/ML_course/projects/project1/data/dataset_to_release", sub_sample=False, num=25000)
    lambda_ = study.best_params['lambda_']
    n_com = study.best_params['n_com']
    gamma = study.best_params['gamma']
    row_nan = 0.5
    # feature_nan = trial.suggest_float('feature_nan', 0.5, 0.7)
    feature_nan = 0.2
    z_threshold = study.best_params['z_threshold']
    feature_threshold = study.best_params['feature_threshold']
    pre_train_data, y_train_processed, x_test_processed = data_processing(x_train1, y_train1, row_nan, feature_nan, z_threshold, feature_threshold, x_test1)
    if PCA:
        x_pca, eig_vec, _, _ = pca(pre_train_data, n_com)
        x_pca = np.real(x_pca)
        x_test_processed = np.real(x_test_processed @ eig_vec)

        sub_x, sub_y = split_cross_validation(x_pca, y_train_processed, 5)
        sub_cur_x = sub_x.copy()
        sub_cur_y = sub_y.copy()
        x_v, y_v = sub_cur_x.pop(1), sub_cur_y.pop(1)
        x_t, y_t = np.vstack(sub_cur_x), np.hstack(sub_cur_y)  

        initial_w = np.random.randn(x_t.shape[1]) * 0.01
        x_t, y_t = data_augmentation(x_t, y_t)
        if HINGE: 
            w, loss, best_f1, best_threshold = hinge_regression(y_t, x_t, y_v, x_v, initial_w, lambda_=lambda_, max_iters=500, gamma=gamma)
        else:
            w, loss, best_f1, best_threshold = reg_logistic_regression(y_t, add_bias(x_t), y_v, add_bias(x_v), lambda_=lambda_, initial_w=initial_w, max_iters=200, gamma=gamma)
    else:
        initial_w = np.random.randn(pre_train_data.shape[1]) * 0.01
        sub_x, sub_y = split_cross_validation(pre_train_data, y_train_processed, 5)
        sub_cur_x = sub_x.copy()
        sub_cur_y = sub_y.copy()
        x_v, y_v = sub_cur_x.pop(1), sub_cur_y.pop(1)
        x_t, y_t = np.vstack(sub_cur_x), np.hstack(sub_cur_y)  
        x_t, y_t = data_augmentation(pre_train_data, y_train_processed)

        if HINGE: 
            w, loss, best_f1, best_threshold = hinge_regression(y_t, x_t, y_v, x_v, initial_w, lambda_=lambda_, max_iters=500, gamma=gamma)
        else:
            w, loss, best_f1, best_threshold = reg_logistic_regression(y_t, add_bias(x_t), y_v, add_bias(x_v), lambda_=lambda_, initial_w=initial_w, max_iters=200, gamma=gamma)

    model = dict(PCA=PCA, HINGE=HINGE, w=w, best_f1=best_f1, best_threshold=best_threshold, losses=loss, 
                 lambda_=lambda_, n_com=n_com, gamma=gamma)
    model_name = "./model_{}".format(datetime.datetime.now().strftime("%m%d%Y_%H_%M_%S"))
    np.savez(model_name, model)
    print('best F1 score is: ', best_f1)
    y_pred = ((x_test_processed @ w) > best_threshold).astype(int)
    y_pred[y_pred == 0] = -1

    return y_pred

def vote(study):

    y_preds = []
    for i in range(11):
        print("+++++++++++++++++++ Vote {} ++++++++++++++++++++++", i)
        y_preds.append(create_model(study))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    y_pred = np.sign(np.sum(y_preds, 0))

    return y_pred

def neural_network(x_train, y_train, x_val, y_val, lr=0.001, epochs=500, batch_size=1024, n_pat=10):
    input_size = x_train.shape[1]
    nn_structure = [input_size, 256, 128, 64, 16, 64, 128, 256, input_size]
    network = NeuralNetwork(layer_sizes=nn_structure, output_activation='linear', loss_function='mse')
    _, loss = network.train(x_train, y_train, x_val, y_val, lr, epochs, batch_size, n_pat, early_stop=False)
    # feature_layer = int(len(nn_structure)-1/2)
    # feature = nn_model.get_feature(x_train, feature_layer)
    return network, loss

def main():
    # thresholds for nans
    row_nan = 0.55
    feature_nan = 0.57
    # threshold for categorical features
    threshold_cat = 10
    # threshold for outliers
    z_threshold=1.95
    feature_threshold=0.19

    sampler = optuna.samplers.TPESampler(multivariate=True, n_ei_candidates=50)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    if HINGE:
        study.optimize(objective_hinge, n_trials=100)
    else:
        study.optimize(objective_log, n_trials=100)
    print(study.best_trial)

    return study

def test_nn():
    global x_train, x_test, y_train, train_ids, test_ids
    row_nan = 0.5
    feature_nan = 0.2
    z_threshold = 2
    feature_threshold = 0.3
    x_train_processed, y_train_processed, x_test_processed = data_processing(x_train, y_train, row_nan, feature_nan, z_threshold, feature_threshold, x_test)
    x_t, y_t, x_v, y_v = split_data(x_train_processed, y_train_processed, 0.8)
    nn_model, loss = neural_network(x_t, x_t, x_v, x_v, lr=0.01, batch_size=256, epochs=500)
    feature = nn_model.get_feature(x_t)
    print('dimension of feature is: ', feature.shape)
    plt.figure()
    plt.plot(loss)
    plt.show()


def OC_SWM():
    # thresholds for nans
    row_nan = 0.55
    feature_nan = 0.57
    # threshold for outliers
    z_threshold=1.95
    feature_threshold=0.19
    pre_train_data, y_train_processed, x_test = data_processing(x_train, y_train, row_nan, feature_nan, z_threshold, feature_threshold, x_test)
    x_pca, eig_vec, eig_val,weight = pca(pre_train_data, 100)
    x_pca = np.real(x_pca)
    sub_x, sub_y = split_cross_validation(x_pca, y_train_processed, 5)
    sub_cur_x = sub_x.copy()
    sub_cur_y = sub_y.copy()
    x_v, y_v = sub_cur_x.pop(1), sub_cur_y.pop(1)
    x_t, y_t = np.vstack(sub_cur_x), np.hstack(sub_cur_y)
    # clf = OneClassSVM(gamma='scale', nu=0.7).fit(x_t[y_t!=1, :].squeeze())
    clf = IsolationForest(random_state=10).fit(x_t[y_t!=1, :].squeeze())
    predcited_scores = clf.score_samples(x_v.squeeze())

    print("roc score: ", metrics.roc_auc_score(y_v, -predcited_scores))
    fpr, tpr, thresholds = metrics.roc_curve(y_v, -predcited_scores)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print("best threshold: ", best_threshold, "; TPR: ", tpr[np.argmax(tpr - fpr)], "; FPR: ", fpr[np.argmax(tpr - fpr)])
    print("best f1 score: ", metrics.f1_score(y_v, -predcited_scores > best_threshold))
    print("best accuracy score: ", metrics.accuracy_score(y_v, -predcited_scores > best_threshold))

if __name__ == '__main__':
    # y_preds = []
    # for i in range(5):
    # study = main()
    # fair = False
    # if fair:
    #     y_pred = vote(study)
    # else:
    #     y_pred = create_model(study)
    # create_csv_submission(test_ids, y_pred, "./submission.csv")
    test_nn()

    # OC_SWM()
    # create_csv_submission(test_ids, np.argmax(np.bincount(y_preds, 1), 1), "./submission.csv")
