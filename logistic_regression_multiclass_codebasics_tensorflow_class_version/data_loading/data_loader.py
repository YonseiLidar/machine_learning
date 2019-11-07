from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np


def data_digits(test_size=0.2):
    
    digits = load_digits()
    
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        test_size=test_size)
    
    y_train_cls = deepcopy(y_train)
    y_test_cls = deepcopy(y_test)
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    data = (x_train, x_test, y_train, y_test, y_train_cls, y_test_cls)
    
    return data 