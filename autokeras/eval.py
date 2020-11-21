import numpy as np
from autokeras.utils import pickle_from_file

if __name__ == '__main__':
    x_train = np.load('path')
    y_train = np.load('path')
    x_test = np.load('path')
    y_test = np.load('path')
    print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')
    print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')
    print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')
    print(f'SHAPE OF TESTING LABELS : {y_test.shape}')

    model = pickle_from_file('path')
    score = model.evaluate(x_test, y_test)
    print(score * 100)







