from utils import (MnistDataset)




if __name__ == "__main__":
    md = MnistDataset()
    X_train, y_train = md.train_dataset()

    print('Rows: {0}, columns: {1}'.format(
            X_train.shape[0], X_train.shape[1]))
    
    X_test, y_test  = md.test_dataset()

    print('Rows: {0}, columns: {1}'.format(
            X_test.shape[0], X_test.shape[1]))