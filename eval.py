import numpy as np
from pyod.models.iforest import IForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import torch
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def attack_model(X_train, X_test, y_test, n_estimators=100, max_samples='auto', max_features=0.5,
                 contamination=0.15, bootstrap=False, n_jobs=-1):
    model = IForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=42
    )
    model.fit(X_train)
    y_train_pred = model.predict(X_train)
    y_train = np.zeros(len(y_train_pred))
    acc_train = accuracy_score(y_train, y_train_pred)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"acc_train: {acc_train:.4f}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1 Score: {f1:.4f}")

def scaler(X_train, X_test, mode='bn'):
    if mode == 'bn':
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    if mode == 'ln':
        X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[3])
        X_test = X_test.reshape(X_test.shape[0], -1, X_test.shape[3])
        X_train_scaled = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-6)
        X_test_scaled = (X_test - np.mean(X_test, axis=1, keepdims=True)) / (np.std(X_test, axis=1, keepdims=True) + 1e-6)
        X_train_scaled = X_train_scaled.reshape(X_train.shape[0], -1)
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], -1)
    if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
        print("warning: NaN or Inf values found in training data, replacing them with 0")
        X_train_scaled = np.nan_to_num(X_train_scaled)
    if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
        print("warning: NaN or Inf values found in training data, replacing them with 0")
        X_test_scaled = np.nan_to_num(X_test_scaled)
    return X_train_scaled, X_test_scaled

def eval(params_dict, mia_data_path='./model_outputs/mia_score.npy', train_num=1000, eval_num=500):
    prenonmem_mem_nonmem_score = np.load(mia_data_path)
    nonmem_score_ = prenonmem_mem_nonmem_score[:train_num]
    evalnonmem_score_ = prenonmem_mem_nonmem_score[-eval_num:]
    evalmem_score_ = prenonmem_mem_nonmem_score[train_num:-eval_num]

    nonmem_score_mean = np.mean(nonmem_score_, axis=2)
    evalnonmem_score_mean = np.mean(evalnonmem_score_, axis=2)
    evalmem_score_mean = np.mean(evalmem_score_, axis=2)
    nonmem_score_std = np.std(nonmem_score_, axis=2)
    evalnonmem_score_std = np.std(evalnonmem_score_, axis=2)
    evalmem_score_std = np.std(evalmem_score_, axis=2)
    nonmem_score_ = np.concatenate((nonmem_score_mean, nonmem_score_std), axis=2)
    evalnonmem_score_ = np.concatenate((evalnonmem_score_mean, evalnonmem_score_std), axis=2)
    evalmem_score_ = np.concatenate((evalmem_score_mean, evalmem_score_std), axis=2)

    X_train = nonmem_score_
    X_test_nonmembers = evalnonmem_score_
    X_test_members = evalmem_score_
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test_nonmembers = X_test_nonmembers.reshape(X_test_nonmembers.shape[0], -1)
    X_test_members = X_test_members.reshape(X_test_members.shape[0], -1)
    y_test = np.array([0] * len(X_test_nonmembers) + [1] * len(X_test_members))
    X_test = np.vstack((X_test_nonmembers, X_test_members))
    print(X_train.shape, X_test.shape, y_test.shape)

    # X_train_scaled, X_test_scaled = scaler(X_train, X_test)
    X_train_scaled = X_train
    X_test_scaled = X_test
    np.random.seed(42)
    torch.manual_seed(42)

    attack_model(
        X_train_scaled,
        X_test_scaled,
        y_test,
        n_estimators=params_dict['n_estimators_values'],
        max_samples=params_dict['max_samples_values'],
        max_features=params_dict['max_features_values'],
        contamination=params_dict['contamination_values'],
        bootstrap=params_dict['bootstrap_values'],
        n_jobs=params_dict['n_jobs']
    )

# Set attack model parameters, you can change them for a better performance
params_dict = {
    'n_estimators_values': 8,
    'max_samples_values': 0.606,
    'max_features_values': 0.396,
    'contamination_values': 0.15,
    'bootstrap_values': False,
    'n_jobs': -1
}
eval(params_dict=params_dict, mia_data_path='./model_outputs/mia_score.npy', train_num=1000, eval_num=500)