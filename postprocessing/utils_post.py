import numpy as np


def normalize_to_01_range(vector):
    min = np.min(vector)
    max = np.max(vector)
    normalized_vector = []
    for x in vector:
        z = (x - min) / (max - min)
        normalized_vector.append(z)
    return np.asarray(normalized_vector)


def normalize_per_dimension(vector_generator):
    max_vector = None
    min_vector = None