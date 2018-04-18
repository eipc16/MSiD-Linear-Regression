# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
import operator as op
from utils import polynomial

def mean_squared_error(x, y, w):
    return np.mean((y - polynomial(x, w))**2)


def design_matrix(x_train, M):
    return np.hstack(map(lambda p: np.power(x_train, p), range(M+1)))


def least_squares(x_train, y_train, M):
    return regularized_least_squares(x_train, y_train, M, 0)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    A = design_matrix(x_train, M)
    B = np.transpose(A)@A
    w = np.linalg.inv(B + regularization_lambda*np.eye(B.shape[0]))@np.transpose(A)@y_train
    return w, mean_squared_error(x_train, y_train, w)

def model_selection(x_train, y_train, x_val, y_val, M_values):
    result = (0, 0, np.inf)
    for i in M_values:
        params, train_err = least_squares(x_train, y_train, M_values[i])
        val_err = mean_squared_error(x_val, y_val, params)
        if val_err < result[2]:
            result = (params, train_err, val_err)
    return result


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    result = (0, 0, np.inf, 0)
    for i in range(len(lambda_values)):
        params, train_err = regularized_least_squares(x_train, y_train, M, lambda_values[i])
        val_err = mean_squared_error(x_val, y_val, params)
        if val_err < result[2]:
            result = (params, train_err, val_err, lambda_values[i])
    return result
