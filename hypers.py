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

def rnn_hypers():
    params = base_formula.copy()
    params['optimizer'] = base_rnn.copy()
    params = _rnn_hypers(params)
    return params


def rnn_hypers_pipeline():
    params = base_pipeline.copy()
    params['optimizer'] = base_rnn.copy()
    params['dataset'] = 'redwine'
    params = _rnn_hypers(params)
    return params


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



def rnn_pipeline():
    rng = np.random
    db = load_db()
    jobs = db.jobs_with(jobset='rnn_hypers', optimizer='rnn')
    jobs = sorted(jobs, key=lambda j:_auc(j['stats']['scores']), reverse=True)
    jobs = jobs[0:20]
    for j in jobs:
        print(j['summary'], j['content'], _auc(j['stats']['scores']))
    job = rng.choice(jobs)
    params = job['content']
    params['dataset'] = 'digits'
    params['grammar'] = 'pipeline'
    params['score'] = {
        'cv': 5
    }
    return params

def _auc(scores):
    x = np.linspace(0, 1, len(scores))
    y = np.maximum.accumulate(scores)
    return auc(x, y)


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
