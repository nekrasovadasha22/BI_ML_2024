import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    y_pred = y_pred.astype(int)
    accuracy = 0
    precision = 0
    recall = 0
    F1 = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            TN += 1
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    return f'accuracy = {accuracy}, precision = {precision}, recall = {recall}, F1 = {F1}'


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    accuracy = correct / len(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    mean_y = np.mean(y_true)
    total_sum = np.sum((y_true - mean_y)**2)
    sq_difference = np.sum((y_true- y_pred)**2)
    r2 = 1 - (sq_difference / total_sum)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    y_pred = np.array(y_pred)
    return np.mean((y_pred - y_true)**2)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true))