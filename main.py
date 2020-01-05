"""
main.py

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

from utils import (
    MnistDataset,
    Pretrained,
    np
)
# from matplotlib import rc
# rc('text', usetex=True)
# rc('font', size=20)

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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


@njit#(parallel=True)
def implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float64)
    # the first dot_product is y0 = 1 *polynomial_expansion(x0 = 0_vect,x)
    v_x[0] = polynomial_expansion(np.zeros(X.shape[1]), x, kernel_degree)
    for k in range(1, v_train_indices.shape[0]):
        xi = X[v_train_indices[k]]
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * polynomial_expansion(xi, x, kernel_degree)

    return v_x


@njit#(parallel=True)
def implicit_form_v(X, v_train_indices, v_label_coeffs):
    v = np.empty(v_train_indices.shape[0], dtype=np.float64)
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
    rl_sum  = 0
    rl = 1
    for i in range(1,c.shape[0]):
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

    rl_sum  = 0
    rl = 1
    for i in range(1,c.shape[0]):
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


def fit(X, y, epoch, kernel_degree):
    return Parallel(n_jobs=4,prefer="threads")(delayed(model)(X, y, i, epoch, kernel_degree) for i in range(10))


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


def test_error(X, models, test, label, kernel_degree):
    scores_random = np.empty(test.shape[0])
    scores_last = np.empty(test.shape[0])
    scores_avg = np.empty(test.shape[0])
    scores_vote = np.empty(test.shape[0])
    j = 0
    for x in test:
        s_random = np.empty(10)
        s_last = np.empty(10)
        s_avg = np.empty(10)
        s_vote = np.empty(10)
        for i in range(10):
            predictions_array = predictions(
                X, models[i, 0], models[i, 1], models[i, 2], x, kernel_degree)
            s_random[i] = predictions_array[0]
            s_last[i] = predictions_array[1]
            s_avg[i] = predictions_array[2]
            s_vote[i] = predictions_array[3]
        # Survival Of The Fittest
        scores_random[j] = highest_score_arg(s_random)
        scores_last[j] = highest_score_arg(s_last)
        scores_avg[j] = highest_score_arg(s_avg)
        scores_vote[j] = highest_score_arg(s_vote)
        j = j + 1

    error_random = np.sum(scores_random != label) / label.shape[0]
    error_last = np.sum(scores_last != label) / label.shape[0]
    error_avg = np.sum(scores_avg != label) / label.shape[0]
    error_vote = np.sum(scores_vote != label) / label.shape[0]

    return error_random, error_last, error_avg, error_vote


def n_mistakes(models):
    m = 0
    for o in range(10):
        m = m + models[o,3]
    return m

def n_supvect(models):
    s_v = 0
    for o in range(10):
        s_v = s_v + models[o,1].shape[0]
    return s_v

def save_models(models, epoch, kernel_degree):
    # print("saving models in models/...")
    pretrained = Pretrained()
    if epoch < 1:
        epoch = '0_{}'.format(int(epoch * 10))
    pretrained.save_model(
        models, 'pretrained_e{0}_k{1}'.format(epoch, kernel_degree))


def load_models(epoch, kernel_degree, same):
    # print("loading models from models/...")
    pretrained = Pretrained()
    if epoch < 1:
        epoch = '0_{}'.format(int(epoch * 10))
    return pretrained.load_model('pretrained_e{0}_k{1}_{2}'.format(epoch, kernel_degree, same))


def train_and_store(X_train, y_train, epoch, kernel_degree):
    models = np.array(fit(X_train, y_train, epoch, kernel_degree))
    save_models(models, epoch, kernel_degree)


def load_and_test(X_train, X_test, y_test, epoch, kernel_degree, same=0):
    models = load_models(epoch, kernel_degree, same)
    e_r, e_l, e_a, e_v = test_error(
        X_train, models, X_test, y_test, kernel_degree)
    perc_r = e_r * 100
    perc_l = e_l * 100
    perc_a = e_a * 100
    perc_v = e_v * 100
    # print("{0:.2f}".format(perc))
    return perc_r, perc_l, perc_a, perc_v


def train_and_store_k_perm(X_train, y_train, epoch, kernel_degree, k):
    np.random.seed(31415)
    print("training k permutation")
    for _ in range(k):
        arr = np.append(X_train, np.expand_dims(y_train, axis=1), axis=1)
        arr = np.random.permutation(arr)
        X_perm = arr[:, 0:-1].copy()
        y_perm = arr[:, -1].copy()
        models = fit(X_perm, y_perm, epoch, kernel_degree)
        save_models(models, epoch, kernel_degree)


def load_and_test_k_perm(X_train, X_test, y_test, epoch, kernel_degree, k):
    print("loading k permutation and training 10 classes")
    for i in range(k):
        models = load_models(epoch, kernel_degree, i)
        error = test_error(X_train, models, X_test, y_test, kernel_degree)
        perc = error * 100
        print("{0:.2f}".format(perc))


def freund_schapire_experiment(X_train, y_train):
    freund_schapire_training(X_train, y_train)
    # TODO
    # freund_schapire_testing(X_test, y_test)


def freund_schapire_training(X_train, y_train):
    print("training the perceptron algorithm on MNIST dataset")

    # from 0.1 to 0.9
    print("epoch: from 0.1 to 0.9")
    for i in range(1, 10):
        for kernel_degree in range(1, 6):
            train_and_store_k_perm(X_train, y_train, i / 10, kernel_degree, 5)

    # from 1 to 9
    print("epoch: from 1 to 9")
    for i in range(1, 10):
        for kernel_degree in range(1, 6):
            train_and_store_k_perm(X_train, y_train, i, kernel_degree, 5)

    # 10 the last width kernel 1
    print("epoch: 10")
    for i in range(10, 11):
        for kernel_degree in range(1, 6):
            train_and_store_k_perm(X_train, y_train, i, kernel_degree, 5)

    # from 20 to 30
    print("epoch: from 20 to 30")
    for i in range(20, 40, 10):
        for kernel_degree in range(2, 6):
            train_and_store_k_perm(X_train, y_train, i, kernel_degree, 5)


def lightweight_training(X_train, y_train):
    print("training the perceptron algorithm on MNIST dataset")

    # from 0.1 to 0.9
    print("epoch: from 0.1 to 0.9")
    for i in tqdm(range(1, 10)):
        for kernel_degree in range(1, 6):
            train_and_store(X_train, y_train, i / 10, kernel_degree)

    # from 1 to 9
    print("epoch: from 1 to 9")
    for i in tqdm(range(1, 10)):
        for kernel_degree in range(1, 6):
            train_and_store(X_train, y_train, i, kernel_degree)

    # 10 the last width kernel 1
    print("epoch: 10")
    for i in tqdm(range(10, 11)):
        for kernel_degree in range(1, 6):
            train_and_store(X_train, y_train, i, kernel_degree)

    # from 20 to 30
    print("epoch: from 20 to 30")
    for i in tqdm(range(20, 40, 10)):
        for kernel_degree in range(2, 6):
            train_and_store(X_train, y_train, i, kernel_degree)


def lightweight_testing(X_train, X_test, y_test):
    print("testing the perceptron algorithm on MNIST dataset")
    errors = []

    for kernel_degree in range(1, 6):
        same_kernel_errors = []

        # from 0.1 to 0.9
        print("epoch: from 0.1 to 0.9")
        for i in tqdm(range(1, 10)):
            same_kernel_errors.append(load_and_test(
                X_train, X_test, y_test, i / 10, kernel_degree))

        # from 1 to 9
        print("epoch: from 1 to 9")
        for i in tqdm(range(1, 10)):
            same_kernel_errors.append(load_and_test(
                X_train, X_test, y_test, i, kernel_degree))

        # 10 the last width kernel 1
        print("epoch: 10")
        for i in tqdm(range(10, 11)):
            same_kernel_errors.append(load_and_test(
                X_train, X_test, y_test, i, kernel_degree))

        # from 20 to 30
        print("epoch: from 20 to 30")
        for i in tqdm(range(20, 40, 10)):
            same_kernel_errors.append(load_and_test(
                X_train, X_test, y_test, i, kernel_degree))
        errors.append(same_kernel_errors)

    return errors


def lightweight_experiment():
    md = MnistDataset()
    # split data
    X_train, y_train = md.train_dataset()

    X_test, y_test = md.test_dataset()

    lightweight_training(X_train, y_train)

    errors = lightweight_testing(X_train, X_test, y_test)

    """ error_random = []
    error_last = []
    error_avg = []
    error_vote = []
    kernel = 4

    print("epoch: from 0.1 to 0.9 kernel:{}".format(kernel))
    x1 = np.arange(0.1, 1, 0.1)
    x2 = np.arange(1, 11)
    for i in tqdm(x1):
        e_r, e_l, e_a, e_v = load_and_test(X_train, X_test, y_test, i, kernel)
        error_random.append(e_r)
        error_last.append(e_l)
        error_avg.append(e_a)
        error_vote.append(e_v)
    print("epoch: from 1 to 10 kernel:{}".format(kernel))
    for i in tqdm(x2):
        e_r, e_l, e_a, e_v = load_and_test(X_train, X_test, y_test, i, kernel)
        error_random.append(e_r)
        error_last.append(e_l)
        error_avg.append(e_a)
        error_vote.append(e_v)

    log_plot(np.concatenate((x1, x2)), error_random, error_last, error_avg, error_vote, kernel) """


def simple_plot(errors, x, kernel_degree):
    plt.style.use('seaborn')
    plt.plot(x, errors, label='last(unorm)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Error')
    plt.title('d={}'.format(kernel_degree))
    plt.legend()
    plt.show()


def log_plot(x, error_random, error_last, error_avg, error_vote, kernel_degree):
    """ errors should contains:
        - error_random,
        - error_last,
        - error_avg,
        - error_vote
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, error_random, label='random(unorm)')
    ax.plot(x, error_last, label='last(unorm)')
    ax.plot(x, error_avg, label='avg(unorm)')
    ax.plot(x, error_vote, label='vote')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title('d={}'.format(kernel_degree))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    md = MnistDataset()
    # split data
    X_train, y_train = md.train_dataset()

    X_test, y_test = md.test_dataset()

    lightweight_training(X_train,y_train)
   