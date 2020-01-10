from utils import(
    MnistDataset,
    tqdm,
    load_and_test,
    log_plot,
    load_models,
    train_and_store
)
import numpy as np

if __name__ == "__main__":

    md = MnistDataset()
    # split data
    X_train, y_train = md.train_dataset()

    X_test, y_test = md.test_dataset()
    
    # change this to True if you don't wont to use pretrained models
    train = False 

    if train:
        kernel = 1
        x1 = np.arange(0.1, 1, 0.1)
        for i in tqdm(x1):
            train_and_store(X_train, y_train, i, kernel)
        x1 = np.arange(1,11)
        for i in tqdm(x1):
            train_and_store(X_train, y_train, i, kernel)
        kernel = 2
        x1 = np.arange(0.1, 1, 0.1)
        for i in tqdm(x1):
            train_and_store(X_train, y_train, i, kernel)
        x1 = np.arange(1,31)
        for i in tqdm(x1):
            train_and_store(X_train, y_train, i, kernel)

    error_random = []
    error_last = []
    error_avg = []
    error_vote = []
    kernel = 1
    try:
        for i in np.arange(0.1, 1, 0.1):
            load_models(i, 1, 0)
            load_models(i, 2, 0)
        for i in np.arange(1, 11):
            load_models(i, 1, 0)
            load_models(i, 2, 0)
        for i in np.arange(11, 31):
            load_models(i, 2, 0)
    except:
        print("Error: are you sure you copied the files in the model dir?")

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
    log_plot(np.concatenate((x1, x2)), error_random,
             error_last, error_avg, error_vote, kernel)

    # now for kernel = 2

    error_random = []
    error_last = []
    error_avg = []
    error_vote = []
    kernel = 2

    print("epoch: from 0.1 to 0.9 kernel:{}".format(kernel))
    x1 = np.arange(0.1, 1, 0.1)
    x2 = np.arange(1, 31)
    for i in tqdm(x1):
        e_r, e_l, e_a, e_v = load_and_test(X_train, X_test, y_test, i, kernel)
        error_random.append(e_r)
        error_last.append(e_l)
        error_avg.append(e_a)
        error_vote.append(e_v)
    print("epoch: from 1 to 30 kernel:{}".format(kernel))
    for i in tqdm(x2):
        e_r, e_l, e_a, e_v = load_and_test(X_train, X_test, y_test, i, kernel)
        error_random.append(e_r)
        error_last.append(e_l)
        error_avg.append(e_a)
        error_vote.append(e_v)
    log_plot(np.concatenate((x1, x2)), error_random,
             error_last, error_avg, error_vote, kernel)
