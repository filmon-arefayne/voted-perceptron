import numpy as np
import os
from tqdm import tqdm

import urllib.request
from urllib.parse import urljoin
import gzip


class MnistDataset:

    def __init__(self):
        self.mnist_path_dir = 'mnist'
        self.datasets_url = 'http://yann.lecun.com/exdb/mnist/'
        self.refresh = False
        #print('Creating mnist directory if not exists...')
        os.makedirs(self.mnist_path_dir, exist_ok=True)
        
    def download_file(self, download_file):

        output_file = os.path.join(self.mnist_path_dir, download_file)

        if self.refresh or not os.path.isfile(output_file):
            print('downloading {0} from {1}'.format(download_file,self.datasets_url))
            url = urljoin(self.datasets_url, download_file)
            download_url(url, output_file)

        return output_file

    def load_mnist(self, train_test='train'):
        """Load MNIST dataset from 'path' """
        labels_path = self.download_file('{}-labels-idx1-ubyte.gz'.format(train_test))
        images_path = self.download_file('{}-images-idx3-ubyte.gz'.format(train_test))
        
        with gzip.open(labels_path, 'rb') as lbpath:
            lbpath.read(8)
            buffer = lbpath.read()
            labels = np.frombuffer(buffer, dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            imgpath.read(16)
            buffer = imgpath.read()
            images = np.frombuffer(buffer, 
                dtype=np.uint8).reshape(len(labels), 
                784).astype(np.float64)
        
        return images, labels

    def train_dataset(self):
        return self.load_mnist(train_test='train')

    def test_dataset(self):
        return self.load_mnist(train_test='t10k')


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_file):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_file, reporthook=t.update_to)
