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

PCA = True
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
    x_train_processed = one_hot_encoding(x_train_processed, cat_indices)
    x_test_processed = one_hot_encoding(x_test_processed, cat_indices)
    x_train_processed, train_mean, train_std = standardize(x_train_processed)
    x_test_processed = (x_test_processed - train_mean) / train_std
    x_train_processed, y_train_processed = z_outlier_removal(x_train_processed, y_train_processed, z_threshold, feature_threshold)

    return x_train_processed, y_train_processed, x_test_processed

def objective_hinge(trial):
    global x_train, x_test, y_train, train_ids, test_ids
    lambda_ = trial.suggest_float('lambda_', 5e-3, 5)
    thres = trial.suggest_float('thres', 0.5, 2)
    n_com = trial.suggest_int('n_com', 10, 200)
    row_nan = trial.suggest_float('row_nan', 0.3, 0.7)
    feature_nan = trial.suggest_float('feature_nan', 0.5, 0.7)
    z_threshold = trial.suggest_float('z_threshold', 1.8, 2.5)
    feature_threshold = trial.suggest_float('feature_threshold', 0.1, 0.3)

    pre_train_data, y_train_processed, x_test_processed = data_processing(x_train, y_train, row_nan, feature_nan, z_threshold, feature_threshold, x_test)
    if PCA:
        x_pca, eig_vec, eig_val,weight = pca(pre_train_data, n_com)
        x_pca = np.real(x_pca)
        x_pca_t = x_pca.copy()
        sub_x, sub_y = split_cross_validation(x_pca_t, y_train_processed, 5)
    else:
        sub_x, sub_y = split_cross_validation(pre_train_data, y_train_processed, 5)
    accs = []
    f1s = []
    # cross-validation
    for i in range(5):
        sub_cur_x = sub_x.copy()
        sub_cur_y = sub_y.copy()
        x_v, y_v = sub_cur_x.pop(i), sub_cur_y.pop(i)
        x_t, y_t = np.vstack(sub_cur_x), np.hstack(sub_cur_y)
        x_t, y_t = data_augmentation(x_t, y_t)
        initial_w = np.random.randn(x_t.shape[1]) * 0.01
        w, loss = hinge_regression(y_t, x_t, initial_w, lambda_=lambda_, max_iters=250, gamma=0.01)
        y_pred = ((x_v @ w) > thres).astype(int)
        accs.append(predict_acc_pure(y_pred, y_v))
        f1s.append(predict_f1_pure(y_pred, y_v))
    print("Average accuracy score is: ", np.mean(accs))
    print("Average f1 score is: ", np.mean(f1s))
    return np.mean(f1s)

def objective_log(trial):
    global x_train, x_test, y_train, train_ids, test_ids
    lambda_ = trial.suggest_float('lambda_', 5e-3, 2)
    thres = trial.suggest_float('thres', 0.5, 0.9)
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
        w, loss = reg_logistic_regression(y_t, x_t, lambda_=lambda_, initial_w=initial_w, max_iters=250, gamma=gamma, verbose=False)
        y_pred = ((x_v @ w) > thres).astype(int)
        accs.append(predict_acc_pure(y_pred, y_v))
        f1s.append(predict_f1_pure(y_pred, y_v))
        losss.append(loss)
    print("Average accuracy score is: ", np.mean(accs))
    print("Average f1 score is: ", np.mean(f1s))
    print("Average loss is: ", np.mean(losss))
    return np.mean(f1s)


def create_model(study):
    x_train1, x_test1, y_train1, train_ids1, test_ids1 = load_csv_data_new("/home/zewzhang/Course/ML/ML_course/projects/project1/data/dataset_to_release", sub_sample=False, num=25000)
    lambda_ = study.best_params['lambda_']
    thres = study.best_params['thres']
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
        x_pca = add_bias(np.real(x_pca))
        x_test_processed = x_test_processed @ eig_vec  
        x_test_processed = add_bias(np.real(x_test_processed))
        initial_w = np.random.randn(x_pca.shape[1]) * 0.01
        x_pca, y_train_processed = data_augmentation(x_pca, y_train_processed)
        w, loss = reg_logistic_regression(y_train_processed, x_pca, lambda_=lambda_, initial_w=initial_w, max_iters=50, gamma=gamma, verbose=True)
    else:
        initial_w = np.random.randn(pre_train_data.shape[1]) * 0.01
        pre_train_data = add_bias(pre_train_data)
        pre_train_data = data_augmentation(pre_train_data)
        w, loss = reg_logistic_regression(y_train_processed, pre_train_data, lambda_=lambda_, initial_w=initial_w, max_iters=50, gamma=gamma, verbose=True)
    y_pred = ((x_test_processed @ w) > thres).astype(int)
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
    study.optimize(objective_log, n_trials=100)
    print(study.best_trial)

    return study

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
    study = main()
    fair = False
    if fair:
        y_pred = vote(study)
    else:
        y_pred = create_model(study)
        
    create_csv_submission(test_ids, y_pred, "./submission.csv")
    # OC_SWM()
    # create_csv_submission(test_ids, np.argmax(np.bincount(y_preds, 1), 1), "./submission.csv")
