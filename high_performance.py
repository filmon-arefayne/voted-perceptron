"""
high_performace.py

    A python file used to represent the voted perceptron

    most common variables
    ----------
    kernel_degree : int
        the degree of the polynomial kernel function

    v_label_coeffs : ndarray
        The label components of the prediction vectors.

    v_train_indices : ndarray
        The training case indices of the prediction vectors.

    c : ndarray
        The votes for the prediction vectors.

    functions
    p.s. most of the function are decorated with njit,
    njit is the nopython mode of numba and it is the
    compiled mode (*the code inside those function
    will be strange and non pythonic*)
    ----------
    """


import numpy as np
from numba import njit, prange
from math import copysign
from tqdm import tqdm
from joblib import Parallel, delayed
import faulthandler

faulthandler.enable()


@njit
def train(X, y, epochs, kernel_degree):
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

            kernel_degree: int
                The number of the degree in the polynomial kernel
            """

    # prediction vector's xi
    # v_train_indices = []
    # prediction vector's labels
    # v_label_coeffs = []
    # weights of the prediction vectors

    # v1 = np.zeros(X.shape[1])
    # don't call np.array on a np.array var
    v_train_indices = np.array([0], dtype=np.int64)
    v_label_coeffs = np.array([0], dtype=np.int64)
    c = np.array([0], dtype=np.int64)
    # they all have value = 0
    # i will not consider the first elements
    weight = 0
    mistakes = 0

    for _ in range(epochs):
        # for xi, label in zip(X, y):
        # numba don't support nested arrays

        for i in range(X.shape[0]):
            xi = X[i]
            label = y[i]
            # same here i can't use sum over the prediction vector
            # we need to iterate over a variable
            # we define a new function
            y_hat = copysign(1, implicit_form_product(
                X, v_train_indices, v_label_coeffs, xi, kernel_degree)[-1])
            # we take always the last prediction vector's product
            # complexity of implicit_form_product is O(k)
            if y_hat == label:
                weight = weight + 1
            else:
                c = np.append(c, np.array([weight]), axis=0)
                v_train_indices = np.append(
                    v_train_indices, np.array([i]), axis=0)
                v_label_coeffs = np.append(
                    v_label_coeffs, np.array([label]), axis=0)
                # reset #C_k+1 = 1
                weight = 1
                mistakes = mistakes + 1

    c = np.append(c, np.array([weight]), axis=0)
    c = c[1:c.shape[0]]
    return v_train_indices, v_label_coeffs, c, mistakes


@njit  # (parallel=True)
def implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float32)
    # the first dot_product is y0 = 1 *polynomial_expansion(x0 = 0_vect,x)
    v_x[0] = polynomial_expansion(np.zeros(X.shape[1], dtype=np.float32), x, kernel_degree)
    for k in range(1, v_train_indices.shape[0]):
        xi = X[v_train_indices[k]]
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * polynomial_expansion(xi, x, kernel_degree)

    return v_x


@njit  # (parallel=True)
def implicit_form_v(X, v_train_indices, v_label_coeffs):
    v = np.empty(v_train_indices.shape[0], dtype=np.float32)
    # v0
    v[0] = 0
    # the first product is y0 = 1 * x0 = 0_vect
    for k in range(1, v_train_indices.shape[0]):
        yi = v_label_coeffs[k]
        xi = X[v_train_indices[k]]
        v[k] = v[k - 1] + (yi * xi)

    return v


@njit
def polynomial_expansion(xi, xj, d):
    return (1 + np.dot(xi, xj)) ** d


# _________________________________________________________________________________
# prediction functions


@njit
def last_unnormalized(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    """Compute score using the final prediction vector(unnormalized)"""
    """ x: unlabeled instance"""
    score = implicit_form_product(X,
                                  v_train_indices, v_label_coeffs, x, kernel_degree)[-1]

    return score


@njit
def normalize(score, v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return score
    return score / norm


@njit
def last_normalized(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    """Compute score using the final prediction vector(normalized)"""
    """ x: unlabeled instance"""
    score = last_unnormalized(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)

    return normalize(score, implicit_form_v(X, v_train_indices, v_label_coeffs)[-1])


@njit
def vote(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using analog of the deterministic leave-one-out conversion"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(X,
                                         v_train_indices, v_label_coeffs, x, kernel_degree)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * copysign(1, v_x)

    return np.sum(s)


@njit
def avg_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using an average of the prediction vectors"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(X,
                                         v_train_indices, v_label_coeffs, x, kernel_degree)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * v_x

    return np.sum(s)


@njit
def avg_normalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using an average of the prediction vectors(normalized)"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(X,
                                         v_train_indices, v_label_coeffs, x, kernel_degree)
    v = implicit_form_v(X, v_train_indices, v_label_coeffs)
    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * normalize(v_x, v[i])

    return np.sum(s)


@njit
def random_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using analog of the randomized leave-one-out 
    method in which we predict using the prediction vectors 
    that exist at a randomly chosen “time slice.”"""
    """ x: unlabeled instance"""
    t = np.sum(c)
    # time slice
    r = np.random.randint(t + 1)
    rl_sum = 0
    rl = 1
    for i in range(1, c.shape[0]):
        if rl_sum > r:
            break
        rl_sum = rl_sum + c[i]
        rl = rl + 1
    rl = rl - 1
    score = implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)[rl]
    return score


@njit
def random_normalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using analog of the randomized leave-one-out 
    method in which we predict using the prediction vectors 
    that exist at a randomly chosen “time slice.(normalized)”"""
    """ x: unlabeled instance"""
    t = np.sum(c)
    # time slice
    # np.random.random_integers(low=0, high=t)  inclusive(low and high)
    # numba doesn't support random_integers
    # randint is exclusive
    r = np.random.randint(t + 1)

    score = implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)

    rl_sum = 0
    rl = 1
    for i in range(1, c.shape[0]):
        if rl_sum > r:
            break
        rl_sum = rl_sum + c[i]
        rl = rl + 1
    rl = rl - 1
    score = implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)[rl]

    return normalize(score, implicit_form_v(X, v_train_indices, v_label_coeffs)[rl])


@njit
def highest_score_arg(s):
    return np.argmax(s)


@njit
def highest_score(s):
    return np.max(s)


# _________________________________________________________________________________
# model functions

# not numba
def fit(X, y, epoch, kernel_degree):
    return Parallel(n_jobs=4, prefer="threads")(delayed(model)(X, y, i, epoch, kernel_degree) for i in range(10))


@njit
def model(X, y, class_type, epoch, kernel_degree):
    y = np.where(y == class_type, 1, -1)
    if epoch < 1:
            # contiguous arrays
        fraction_x = X[0:int(X.shape[0] * epoch),
                       :].copy()
        fraction_y = y[0:int(X.shape[0] * epoch)].copy()
        return train(fraction_x, fraction_y, 1, kernel_degree)
    return train(X, y, epoch, kernel_degree)


@njit
def predictions(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    s_random = random_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree)
    s_last = last_unnormalized(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)
    s_avg = avg_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree)
    s_vote = vote(X, v_train_indices, v_label_coeffs, c, x, kernel_degree)

    return np.array([s_random, s_last, s_avg, s_vote])


# _________________________________________________________________________________
# Kernel Matrix version (Gram)

# for the kernel matrix
@njit
def gram_build(X, kernel_degree):
    Gram = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(Gram.shape[0]):
        for j in range(i, Gram.shape[0]):
            if i <= j:
                Gram[i, j] = polynomial_expansion(X[i], X[j], kernel_degree)
                Gram[j, i] = Gram[i, j]
    return Gram


def gram_fit(X, y, epoch, kernel_degree):
    return Parallel(n_jobs=2, prefer="threads")(delayed(gram_model)(X, y, i, epoch, kernel_degree) for i in range(10))


@njit
def gram_model(X, y, class_type, epoch, kernel_degree):
    y = np.where(y == class_type, 1, -1)
    if epoch < 1:
            # contiguous arrays
        fraction_x = X[0:int(X.shape[0] * epoch),
                       :].copy()
        fraction_y = y[0:int(X.shape[0] * epoch)].copy()
        return gram_train(fraction_x, fraction_y, 1, kernel_degree)
    return gram_train(X, y, epoch, kernel_degree)


@njit
def gram_predictions(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    s_random = gram_random_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index)
    s_last = gram_last_unnormalized(
        X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)
    s_avg = gram_avg_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index)
    s_vote = gram_vote(X, v_train_indices, v_label_coeffs,
                       c, x, kernel_degree, gram_index)

    return np.array([s_random, s_last, s_avg, s_vote])


@njit
def gram_train(X, y, epochs, kernel_degree):
    v_train_indices = np.array([0], dtype=np.int64)
    v_label_coeffs = np.array([0], dtype=np.int64)
    c = np.array([0], dtype=np.int64)
    weight = 0
    mistakes = 0

    for _ in range(epochs):
        for i in range(X.shape[0]):
            xi = X[i]
            label = y[i]

            y_hat = copysign(1, gram_implicit_form_product(
                X, v_train_indices, v_label_coeffs, xi, kernel_degree, i)[-1])
            if y_hat == label:
                weight = weight + 1
            else:
                c = np.append(c, np.array([weight]), axis=0)
                v_train_indices = np.append(
                    v_train_indices, np.array([i]), axis=0)
                v_label_coeffs = np.append(
                    v_label_coeffs, np.array([label]), axis=0)
                weight = 1
                mistakes = mistakes + 1

    c = np.append(c, np.array([weight]), axis=0)
    c = c[1:c.shape[0]]
    return v_train_indices, v_label_coeffs, c, mistakes


@njit
def gram_implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float32)
    v_x[0] = polynomial_expansion(
        np.zeros(X.shape[1], dtype=np.float32), x, kernel_degree)
    assert('Gram_train' in globals())
    for k in range(1, v_train_indices.shape[0]):
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * Gram_train[gram_index, v_train_indices[k]]

    return v_x


@njit
def gram_test_implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float32)
    v_x[0] = polynomial_expansion(
        np.zeros(X.shape[1], dtype=np.float32), x, kernel_degree)
    assert('Gram_test' in globals())
    for k in range(1, v_train_indices.shape[0]):
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * Gram_test[gram_index, v_train_indices[k]]

    return v_x


@njit
def gram_last_unnormalized(X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index):
    """Compute score using the final prediction vector(unnormalized)"""
    """ x: unlabeled instance"""
    score = gram_test_implicit_form_product(X,
                                            v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)[-1]

    return score


@njit
def gram_vote(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    """Compute score using analog of the deterministic leave-one-out conversion"""
    """ x: unlabeled instance"""

    dot_products = gram_test_implicit_form_product(X,
                                                   v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * copysign(1, v_x)

    return np.sum(s)


@njit
def gram_avg_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    """Compute score using an average of the prediction vectors"""
    """ x: unlabeled instance"""

    dot_products = gram_test_implicit_form_product(X,
                                                   v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * v_x

    return np.sum(s)


@njit
def gram_random_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    """Compute score using analog of the randomized leave-one-out 
    method in which we predict using the prediction vectors 
    that exist at a randomly chosen “time slice.”"""
    """ x: unlabeled instance"""
    t = np.sum(c)
    # time slice
    r = np.random.randint(t + 1)
    rl_sum = 0
    rl = 1
    for i in range(1, c.shape[0]):
        if rl_sum > r:
            break
        rl_sum = rl_sum + c[i]
        rl = rl + 1
    rl = rl - 1
    score = gram_test_implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)[rl]
    return score
