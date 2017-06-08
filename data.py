import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

from formula import grammar, _eval

autoweka = (
    "dexter",
    "germancredit",
    "dorothea",
    "yeast",
    "amazon",
    "secom",
    "semeion",
    "car",
    "madelon",
    "krvskp",
    "abalone",
    "winequalitywhite",
    "waveform",
    "gisette",
    "convex",
    "cifar10small",
    "mnist",
    "mnistrotationbackimagenew",
    "shuttle",
    "kddcup09appetency",
    "cifar10",
)

def get_dataset(name, which='train'):
    if name == 'digits':
        dataset = datasets.load_digits()
        X = dataset['data']
        y = dataset['target']
        return _split(X, y)
    elif name == 'redwine':
        df = pd.read_csv('uci_standard/winequality-red.csv')
        data = df.values
        X, y = data[:, 0:-1], data[:, -1]
        y = np.int32(y)
        return _split(X, y)
    elif name in autoweka:
        X, y = _loadarff('autoweka/{}/train.arff'.format(name))
        X_train, X_test, y_train, y_test = _split(X, y)
        X_train_orig = X_train
        X_train = _preprocess(X_train, X_train=X_train, name=name)
        X_test = _preprocess(X_test, X_train=X_train, name=name)
        if which =='train':
            return X_train, X_test, y_train, y_test
        elif which == 'test':
            X, y = _loadarff('autoweka/{}/test.arff'.format(name))
            X = _preprocess(X, X_train=X_train_orig, name=name)
            return X, y
        else:
            raise ValueError('wrong which')
    elif name == "iris":
        dataset = datasets.load_iris()
        X = dataset['data']
        y = dataset['target']
        return _split(X, y)
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


def _split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


def _preprocess(X, X_train, name):
    if name == "secom":
        X = np.array(X, dtype='float64')
        X = Imputer(missing_values='NaN', strategy='median', verbose=1).fit_transform(X_train)
    elif name in ("abalone", "germancredit", "car", "krvskp"):
        X = pd.DataFrame.from_records(X)
        X = pd.get_dummies(X, columns=None)
        for col in X.columns:
            assert str(X[col].dtype) in ('float64', 'uint8'), X[col].dtype
        X = np.array(X.values, dtype='float64')
    else:
        X = np.array(X, dtype='float64')
    assert not np.any(np.isnan(X))
    assert str(X.dtype) == 'float64'
    return X


def _loadarff(filename):
    data, _ = loadarff(filename)
    df = pd.DataFrame.from_records(data)
    df = df.values
    X =  df[:, 0:-1]
    y = df[:, -1]
    y = _encode(y)
    return X, y


def _encode(y):
    y_uniq = np.unique(y)
    y_uniq = y_uniq.tolist()
    y_uniq = sorted(y_uniq)
    label_to_int = {val: i for i, val in enumerate(y_uniq)}
    y = [label_to_int[val] for val in y]
    y = np.array(y)
    return y

