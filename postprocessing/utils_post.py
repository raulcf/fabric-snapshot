import numpy as np


def normalize_to_01_range(vector):
    min = np.min(vector)
    max = np.max(vector)
    normalized_vector = []
    for x in vector:
        z = (x - min) / (max - min)
        normalized_vector.append(z)
    return np.asarray(normalized_vector)


def normalize_per_dimension(vector, max_vector=None, min_vector=None):
    num = np.asarray(vector) - np.asarray(min_vector)
    den = np.asarray(max_vector) - np.asarray(min_vector)
    norm = num / den
    return norm
