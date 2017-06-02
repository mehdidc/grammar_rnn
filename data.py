import pandas as pd
import numpy as np
from sklearn import datasets
from formula import grammar, _eval

def get_dataset(name):
    from sklearn import datasets

    if name == 'digits':
        dataset = datasets.load_digits()
        X = dataset['data']
        y = dataset['target']
        return X, y
    elif name == 'redwine':
        df = pd.read_csv('uci_standard/winequality-red.csv')
        data = df.values
        X, y = data[:, 0:-1], data[:, -1]
        y = np.int32(y)
        return X, y
    elif name == "iris":
        dataset = datasets.load_iris()
        X = dataset['data']
        y = dataset['target']
        return X, y
    else:
        try:
            grammar.parse(name)
        except Exception as ex:
            raise ValueError(name, ex)
        else:
            rng = np.random.RandomState(42)
            X = rng.uniform(-3, 3, size=1000)
            y  = _eval(name, X) 
            return X, y
