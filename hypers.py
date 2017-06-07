import os
import random

import pandas as pd
from sklearn.metrics import auc
import numpy as np
from lightjob.cli import load_db

def generate_job(jobset):
    func = globals()[jobset]
    return func()

base_formula = {
    'dataset': 'x*x+sin(x)*cos(x)',
    'grammar': 'formula',
    'optimizer': {
    },
    'score': {
        'thresh': 0.1
    },
    'random_state': 42
}

base_pipeline = {
    'dataset': 'redwine',
    'grammar': 'pipeline',
    'optimizer': {
    },
    'score': {
        'cv' : 5,
        'scoring': 'accuracy'
    },
    'random_state': 42
}

base_random = {
    'name': 'random',
    'params': {
        'min_depth': 1,
        'max_depth': 5,
        'strict_depth_limit': False,
        'nb_iter': 10,
    }
}

base_rnn = {
    'name': 'rnn',
    'params': {
        'min_depth': 1,
        'max_depth': 5,
        'strict_depth_limit': False,
        'nb_iter': 10,
        'emb_size': 128,
        'hidden_size': 128,
        'algo':{
            'name': 'sgd',
            'params': {
                'lr': 1e-3,
            }
        },
        'init_ih_std': 0.08,
        'init_hh_std': 0.08,
        'gamma': 0.9
    }
}


# in datasets was meant to represent the set of datasets
# where we fit an RNN in order to use the RNN in the
# out datasets
in_datasets = (
    #"dexter",
    #"germancredit",
    #"dorothea",
    "yeast",
    "amazon",
    "secom",
    "semeion",
    #"car",
    #"madelon",
    #"krvskp",
    "abalone",
    "winequalitywhite",
    "waveform",
)
out_datasets = (
    "convex",
)

def pipeline():
    func = random.choice((rnn_pipeline, random_pipeline))
    return func()

def rnn_pipeline():
    params = base_pipeline.copy()
    params['optimizer'] = {
        'name': 'rnn',
        'params':{
            'min_depth': 1,
            'max_depth': 5,
            'strict_depth_limit': False,
            'nb_iter': 100,
            'emb_size': 128,
            'hidden_size': 32,
            'nb_layers': 2,
            'algo': {
                'name': 'adam',
                'params':{
                    'lr': 1e-3
                }
            },
            'gamma': 0.9,
            'init_ih_std': 0.08,
            'init_hh_std': 0.08
        }
    }
    dataset = random.choice(in_datasets)
    params['dataset'] = dataset
    params['grammar'] = 'pipeline'
    params['score'] = base_pipeline['score']
    params['random_state'] = _random_state()
    return params


def random_pipeline():
    params = base_pipeline.copy()
    params['optimizer'] = base_random.copy()
    params['optimizer']['params']['nb_iter'] = 100
    dataset = random.choice(in_datasets)
    params['dataset'] = dataset
    params['grammar'] = 'pipeline'
    params['random_state'] = _random_state() 
    return params

# OUT
def out_pipeline():
    func = random.choice((out_random_pipeline, out_frozen_rnn_pipeline))
    return func()

# OUT Random
def out_random_pipeline():
    params = base_pipeline.copy()
    params['optimizer'] = base_random.copy()
    params['optimizer']['params']['nb_iter'] = 100
    params['dataset'] = random.choice(out_datasets)
    params['grammar'] = 'pipeline'
    params['random_state'] = _random_state() 
    return params

# OUT Frozen RNN
def out_frozen_rnn_pipeline():
    params = base_pipeline.copy()
    params['optimizer'] = {
        'name': 'frozen_rnn',
        'params': {
            'model': 'models/model.th',
            'min_depth': 1,
            'max_depth': 5,
            'strict_depth_limit': False,
            'nb_iter': 100
        }
    }
    params['dataset'] = random.choice(out_datasets)
    params['grammar'] = 'pipeline'
    params['random_state'] = _random_state() 
    return params


def _monotonicity(scores):
    scores = np.array(scores)
    avg = pd.ewma(pd.Series(scores[0:-1]), span=1./0.1-1)
    return (scores[1:] - avg).mean()


def _auc(scores):
    x = np.linspace(0, 1, len(scores))
    y = np.maximum.accumulate(scores)
    return auc(x, y)


def _corr(scores):
    x = np.linspace(0, 1, len(scores))
    y = scores
    c = np.corrcoef(x, y)[0, 1]
    return c


def _time_to_reach(scores, val):
    scores = np.array(scores)
    better = (scores > val)
    if len(scores[better]) == 0:
        return float('inf')
    else:
        return better.argmax()


def _rnn_hypers(params):
    rng = np.random
    rnn = params['optimizer']
    opt = rnn['params']
    opt['nb_iter'] = 1000
    opt['hidden_size'] = rng.randint(5, 256)
    opt['algo'] = {
        'name': rng.choice(('sgd', 'adam')),
        'params': {
            'lr': 10 ** rng.uniform(-5, -1),
        }
    }
    opt['init_ih_std'] = 10 ** rng.uniform(-5, -1)
    opt['init_hh_std'] = 10 ** rng.uniform(-5, -1)
    opt['gamma'] = np.random.uniform(0., 1.)
    params['optimizer'] = rnn
    params['random_state'] = _random_state() 
    return params

def test_formula_random():
    params = base_formula.copy()
    params['optimizer'] = base_random
    return params

def test_formula_rnn():
    params = base_formula.copy()
    params['optimizer'] = base_random
    return params

def test_pipeline_random():
    params = base_pipeline.copy()
    params['optimizer'] = base_random
    return params

def test_pipeline_rnn():
    params = base_pipeline.copy()
    params['optimizer'] = base_rnn
    return params

def _random_state():
    return np.random.randint(0, 1000000000)
