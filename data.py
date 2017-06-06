import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

from sklearn import datasets
from sklearn.model_selection import train_test_split

from formula import grammar, _eval

autoweka = (
    "mnist",
    "germancredit",
    "winequalitywhite",
    "mnistrotationbackimagenew",
    "shuttle",
    "amazon",
    "waveform",
    "convex",
    "cifar10small",
    "gisette",
    "semeion",
    "zip",
    "car",
    "cifar10",
    "yeast",
    "dorothea",
    "abalone",
    "dexter",
    "whitewine",
    "madelon",
    "secom",
    "krvskp",
    "kddcup09appetency",
)

def get_dataset(name):
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
        return _split(X, y)
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

