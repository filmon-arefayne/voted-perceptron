"""
    A python file used to represent the voted perceptron

    most common variables
    ----------
    kernel_degree : int
        the degree of the polynomial kernel function

    v_label_coeffs : ndarray
        The label components of the prediction vectors.

    v_train_terms : ndarray
        The training case components of the prediction vectors.

    c : ndarray
        The votes for the prediction vectors.

    functions
    p.s. most of the function are decorated with njit,
    njit is the nopython mode of numba and it is the
    compiled mode (*the code inside those function
    will be strange and non pythonic*)
    ----------
    """

from utils import (MnistDataset, np)
import matplotlib.pyplot as plt
from numba import njit, prange
from math import copysign
from tqdm import tqdm

@njit
def train(X, y, epoch):
    """Train the voted perceptron.

            Parameters
            ----------
            X : ndarray
                An ndarray where each row is a training case of the mnist dataset.

            y : ndarray
                An ndarray where each element is the label/classification of a
                training case in train_set for binary classification.
                Valid label values are -1 and 1.

            epochs : int
                The number of epochs to train.
            """

    # prediction vector's xi
    # v_train_terms = []
    # prediction vector's labels
    # v_label_coeffs = []
    # weights of the prediction vectors
    c = np.array([1])

    #v1 = np.zeros(X.shape[1])
    v_train_terms = np.zeros((1,X.shape[1])) # don't call np.array on a np.array var
    v_label_coeffs = np.array((1,0))
    # first v = 1*zero_vector
    weight = 0
    for _ in range(epoch):
        # for xi, label in zip(X, y):
        # numba don't support nested arrays
        for i in range(X.shape[0]):
            xi = X[i]
            label = y[i]
            # same here i can't use sum over the prediction vector
            # we need to iterate over a variable
            # we specify a new function
            y_hat = copysign(1, implicit_form_product(v_train_terms, v_label_coeffs, xi)[-1])
            # we take always the last prediction vector's product
            if y_hat == label:
                weight = weight + 1
            else:
                c = np.append(c, np.array([weight]), axis=0)
                v_train_terms = np.append(v_train_terms, np.expand_dims(xi, axis=0), axis=0)
                v_label_coeffs = np.append(v_label_coeffs, np.array([label]), axis=0)
                # reset #C_k+1 = 1
                weight = 1
    c = np.append(c, np.array([weight]), axis=0)
    c = c[1:c.shape[0]-1] # TODO i need to fix this!
    return v_train_terms, v_label_coeffs, c

@njit
def implicit_form_product(v_train_terms, v_label_coeffs, x):
    dot_products = np.empty(v_train_terms.shape[0], dtype=np.float64)
    v_x = np.empty(v_train_terms.shape[0], dtype=np.float64)
    for i in range(v_train_terms.shape[0]):
        xi = v_train_terms[i]
        yi = v_label_coeffs[i]
        dot_products[i] = yi * polynomial_expansion(xi, x)
        if i == 0:
            v_x[0] = dot_products[0]
        else:
            v_x[i] = v_x[i-1] + dot_products[i]

    return v_x

# TODO reduce the number of for loops
@njit
def implicit_form_v(v_train_terms, v_label_coeffs):
    products = np.empty(v_train_terms.shape[0], dtype=np.float64)
    for i in range(v_train_terms.shape[0]):
        xi = v_train_terms[i]
        yi = v_label_coeffs[i]
        products[i] = (yi * xi)
    # i can't use itertools.accumulate and
    # np.add.accumulate(product)
    # is not implemented in numba
    # numba pull #4578
    # we will iterate to create the array
    v = np.empty(v_train_terms.shape[0], dtype=np.float64)
    for i in range(1, products.shape[0]):
        v[i] = v[i - 1] + products[i]

    return v

@njit
def last_unnormalized(v_train_terms, v_label_coeffs, x):
    """Compute score using the final prediction vector(unnormalized)"""
    """ x: unlabeled instance"""
    score = implicit_form_product(v_train_terms,v_label_coeffs, x)[-1]

    return score

@njit
def normalize(score, v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return score
    return score / norm

@njit
def last_normalized(v_train_terms, v_label_coeffs, x):
    """Compute score using the final prediction vector(normalized)"""
    """ x: unlabeled instance"""
    score = last_unnormalized(v_train_terms, v_label_coeffs, x)

    return normalize(score, implicit_form_v(v_train_terms, v_label_coeffs))

#@njit
def vote(v_train_terms, v_label_coeffs, c, x):
    """Compute score using analog of the deterministic leave-one-out conversion"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(v_train_terms, v_label_coeffs, x)

    s = sum(
        weight
        * copysign(1, v_x)
        for weight, v_x
        in zip(c, dot_products)
    )

    return copysign(1, s)


def mnist_train_test(X, y, epoch):
    array = []
    for i in tqdm(range(10)):
        array.append(model(X, y, i, epoch))
        print("number of support vectors:",array[i][0].shape[0])
    return np.array(array)

@njit
def model(X, y, class_type, epoch):
    y = np.where(y == class_type, 1, -1)

    return train(X, y, epoch)

@njit
def highest_score_arg(s):
    return np.argmax(s)

def highest_score(s):
    return np.max(s)

#@njit
def test_error(models, test, label):
    scores = np.empty(test.shape[0])
    j = 0
    for x in test:
        s = np.empty(10)
        for i in range(10):
            s[i] = last_unnormalized(models[i][0], models[i][1], x)
        scores[j] = highest_score_arg(s)
        j = j + 1
    error = (scores != label).sum()
    return error

@njit
def polynomial_expansion(xi, xj, d=5):
    return (1 + np.dot(xi, xj)) ** d


if __name__ == "__main__":
    md = MnistDataset()
    X_train, y_train = md.train_dataset()

    X_test, y_test = md.test_dataset()

    print("training the perceptron algorithm on MNIST dataset")
    models = mnist_train_test(X_train, y_train, 1)
    print("testing the perceptron algorithm on MNIST dataset")
    error = test_error(models, X_test, y_test)
    print(error)

