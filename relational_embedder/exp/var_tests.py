import time
import numpy as np
from scipy.spatial.distance import cosine


def compare_vector_dot_product():

    data_large = [np.asarray([el for el in range(300)]) for _ in range(50000)]

    data_medium = [np.asarray([el for el in range(150)]) for _ in range(50000)]

    data_small = [np.asarray([el for el in range(50)]) for _ in range(50000)]

    st = time.time()
    q = data_large[0]
    dum = 0
    for el in data_large:
        dum += cosine(el, q)

    et = time.time()

    total = et - st
    print("Large took: " + str(total))

    st = time.time()
    q = data_medium[0]
    dum = 0
    for el in data_medium:
        dum += cosine(el, q)
    et = time.time()

    total = et - st
    print("Medium took: " + str(total))

    st = time.time()
    q = data_small[0]
    dum = 0
    for el in data_small:
        dum += cosine(el, q)
    et = time.time()

    total = et - st
    print("Small took: " + str(total))


if __name__ == "__main__":
    print("Test various")

    compare_vector_dot_product()

