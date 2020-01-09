# voted-perceptron

In this repo , we try to reproduce some of the results reported in section 5 of the Freund and Schapire 1999 article (in particular the graphs for d = 1 and d = 2 in Figure 2).

### Paper implemented  
| URL |  Title | 
| --- |  ----- |
|[**10.1023/A:1007662407062**](https://link.springer.com/content/pdf/10.1023/A:1007662407062.pdf) | **Large Margin Classification Using the Perceptron Algorithm** |

## Getting Started
There are two method to run this project:
the preferred method **is to run the colab notebook** the instruction are in it.

These instructions below will get you a copy of the project up and running on your local machine.

### Prerequisites(Local)

What things you need to install the software and how to install them

You will need a Python3 env here i will list the instruction for the **Anaconda** distribution 

## Installation Guide

Navigate in the **Anaconda Prompt** and select your environment for example:
`conda activate pyenv`

To download the repository:

`git clone https://github.com/Filmon97/voted-perceptron.git`

Then you need to install the basic dependencies to run the project on your system:

`pip install -r requirements.txt`


To get the pretrained models you will need to fetch the data from the submodule:

`cd voted-perceptron`

`git submodule init`

`git submodule update`

**Now you to create a 'model' folder and then copy the content of the 'pretrained' folder in it.**

## Built With
* [Numpy](https://numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 
* [Numba](http://numba.pydata.org/) - Numba makes Python code fast.
* [Joblib](https://joblib.readthedocs.io/en/latest/) - Easy simple parallel computing.


## Acknowledgments

* I would like to thank [Fred Foo](https://stackoverflow.com/questions/17720151/how-to-speed-up-kernelize-perceptron-using-parallelization) for suggesting how to compute the Gram Matrix that i used for the [Google Colab](https://colab.research.google.com/) and [dough](https://stackoverflow.com/questions/9478791/is-there-an-enhanced-numpy-scipy-dot-method) for explaining me about Scipy dot method.

