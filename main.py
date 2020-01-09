from utils import(
    MnistDataset,
    tqdm,
    load_and_test,
    log_plot
)
import numpy as np

if __name__ == "__main__":
    
    md = MnistDataset()
    # split data
    X_train, y_train = md.train_dataset()

    X_test, y_test = md.test_dataset()

    error_random = []
    error_last = []
    error_avg = []
    error_vote = []
    # change the kernel to from 1 to 2
    kernel = 1

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
    log_plot(np.concatenate((x1, x2)), error_random, error_last, error_avg, error_vote, kernel) 