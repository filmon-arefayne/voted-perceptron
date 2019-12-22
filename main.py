from utils import (MnistDataset, np)
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

########## voted perceptron ############

@njit
def train(X, y, epoch):
    v1 = np.zeros(X.shape[1])
    c1 = 0
    v = []
    c = []
    v.append(v1)
    c.append(c1)
    # k = 0, v[0] = 0, c[0] = 0
    for _ in range(epoch):
        k = 0
        #for xi, label in zip(X, y):
        # numba don't support nested arrays
        for i in range(len(y)):
            xi = X[i]
            label = y[i] 
            if label == gamma(xi, v[k]):
                c[k] = c[k] + 1
            else:
                v.append(v[k] + label * xi)
                c.append(1)
                k = k + 1
    return (v, c)

@njit
def gamma(xi, vk):
    dot_product = np.dot(xi, vk)
    return np.where(dot_product >= 0.0, 1, -1)


def predict(v, c, x):
    """ x: unlabeled instance"""
    s = 0
    for vi, ci in zip(v, c):
        s = s + ci * gamma(x, vi)
    return np.where(s >= 0.0, 1, -1)


def mnist_train(X, y, epoch):
    print("training the perceptron algorithm on MNIST dataset")
    print("{} elements".format(X.shape[0]))
    print("{} epochs".format(epoch))
    print("#####################################################")
    v = []
    for i in range(10):
        v.append(model(X, y, i, epoch))

    return v


def model(X, y, class_type, epoch):
    print('running one against all for the {} class'.format(class_type))

    y = np.where(y == class_type, 1, -1)

    return train(X, y, epoch)

    #predicted_label = predict(v, c, df_test.iloc[3,:])
    #true_label = y_test[3]
    #print("Predicted label: {}".format("0" if predicted_label == 1 else "not zero"))
    #print("True label: {}".format(true_label))


def highest_score(s):
    return np.argmax(np.array(s))


def last_unnormalized(v, x):
    score = np.dot(v[-1], x)
    # v[-1] is the last perceptron
    return score


def test_error(v, test, label):
    scores = []
    for x in test:
        s = []
        for i in range(10):
            s.append(last_unnormalized(v[i][0], x))
        scores.append(highest_score(s))
    error = (scores != label).sum()
    return error

def kernel_function(xi,xj):
    return 1 + np.dot(xi,xj)

if __name__ == "__main__":
    md = MnistDataset()
    X_train, y_train = md.train_dataset()

    X_test, y_test = md.test_dataset()

    """ df_train = pd.DataFrame(X_train, index=range(X_train.shape[0]),
                            columns=range(X_train.shape[1]))
    df_train_label = pd.DataFrame(y_train, index=range(y_train.shape[0]))

    df_test = pd.DataFrame(X_test, index=range(X_test.shape[0]),
                           columns=range(X_test.shape[1]))
    df_test_label = pd.DataFrame(y_test, index=range(y_test.shape[0]))

    random_test = np.random.choice(range(100))

    first_image = df_test.iloc[random_test, :]
    first_label = y_test[random_test]
    # print(first_image)
    # print(first_label)

    # 784 columns correspond to 28x28 image
    plottable_image = np.reshape(first_image.values, (28, 28))
    # Plot the image
    plt.imshow(plottable_image, cmap='gray_r')
    plt.title('Digit Label: {}'.format(first_label))

    plt.show()

    v = mnist_train(X=df_train.iloc[0:2000, :].values,
                    y=df_train_label.iloc[0:2000, :].values, epoch=2)
    print(test_error(v, df_test.iloc[0:200, :].values, y_test[0:200])) """

    epochs = range(1,10)
    test_errors = []
    for i in epochs:
        v = mnist_train(X_train[0:2000, :], y_train[0:2000], i)
        test_errors.append(test_error(v, X_test[0:200, :], y_test[0:200]))
    
    plt.plot(epochs, test_errors)