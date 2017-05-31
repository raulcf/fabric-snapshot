import numpy as np


def normalize_to_01_range(vector):
    min = np.min(vector)
    max = np.max(vector)
    normalized_vector = []
    for x in vector:
        z = (x - min) / (max - min)
        normalized_vector.append(z)
    return np.asarray(normalized_vector)


def normalize_to_unitrange_per_dimension(vector, max_vector=None, min_vector=None):
    num = np.asarray(vector) - np.asarray(min_vector)
    den = np.asarray(max_vector) - np.asarray(min_vector)
    norm = [0] * len(vector)
    for idx in range(len(vector)):
        if den[idx] == 0:
            norm[idx] = 0
        else:
            norm[idx] = num[idx] / den[idx]
    return np.asarray(norm)


def normalize_per_dimension(vector, mean_vector=None, std_vector=None):
    num = np.asarray(vector) - np.asarray(mean_vector)
    norm = [0] * len(vector)
    for idx in range(len(vector)):
        if std_vector[idx] == 0:
            norm[idx] = 0
        else:
            norm[idx] = num[idx] / std_vector[idx]
    return np.asarray(norm)
