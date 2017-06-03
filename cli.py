import json
import math
from functools import partial
import warnings
import logging
from datetime import datetime
from collections import OrderedDict
from collections import namedtuple
from collections import defaultdict

from clize import run

import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import sklearn
import sklearn.kernel_approximation
import sklearn.naive_bayes
import sklearn.cluster
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform

from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.random import RandomWalker
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.types import Int, Float
from grammaropt.rnn import optimize as rnn_optimize
from grammaropt.random import optimize as random_optimize


from lightjob.cli import load_db
from lightjob.utils import summarize
import random

import pipeline
import formula

from hypers import generate_job
from data import get_dataset
from hypers import _auc, _monotonicity

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
#log.setLevel('DEBUG')
#hndl = logging.FileHandler('log',  mode='w')
#hndl.setLevel('DEBUG')
#log.addHandler(hndl)
hndl = logging.StreamHandler()
log.addHandler(hndl)


grammars = {
    'pipeline': pipeline, 
    'formula': formula
}


def optim(jobset):
    optimizers = {'rnn': _optim_rnn_from_params, 'random': _optim_random_from_params}
    params = generate_job(jobset=jobset)
    print(json.dumps(params, indent=2))
    dataset = params['dataset']
    optimizer = params['optimizer']['name']
    grammar = params['grammar']
    random_state = params['random_state']
    np.random.seed(random_state)

    _optim = optimizers[optimizer]
    start_time = datetime.now()
    stats = _optim(params)
    end_time = datetime.now()
    
    db = load_db()
    db.safe_add_job(
        params, 
        stats=stats, 
        jobset=jobset, 
        dataset=dataset, 
        grammar=grammar, 
        optimizer=optimizer,
        start=start_time,
        end=end_time)


def _optim_random_from_params(params):
    dataset = params['dataset']
    random_state = params['random_state']

    opt = params['optimizer']['params']
    min_depth = opt['min_depth']
    max_depth = opt['max_depth']
    strict_depth_limit = opt['strict_depth_limit']
    nb_iter = opt['nb_iter']

    mod = grammars[params['grammar']]
    grammar = mod.grammar

    X, y = get_dataset(dataset)
    
    score_func = partial(mod.score, X=X, y=y, **params['score'])

    wl = RandomWalker(
        grammar=grammar, 
        min_depth=min_depth, 
        max_depth=max_depth, 
        strict_depth_limit=strict_depth_limit,
        random_state=random_state
    )
    codes, scores = random_optimize(
        score_func, 
        wl, 
        nb_iter=nb_iter
    )
    stats = {'codes': codes, 'scores': scores}
    return stats


def _optim_rnn_from_params(params):
    dataset = params['dataset']
    random_state = params['random_state']
    
    opt = params['optimizer']['params']
    min_depth = opt['min_depth']
    max_depth = opt['max_depth']
    strict_depth_limit = opt['strict_depth_limit']
    nb_iter = opt['nb_iter']

    mod = grammars[params['grammar']]
    grammar = mod.grammar
    rules = mod.rules
    rules = list(rules)
    def key(r):
        if r.name != '':
            return r.name
        elif r.__class__.__name__ == "Literal":
            return r.literal
        elif r.__class__.__name__ == "Sequence":
            return str(tuple(key(m) for m in r.members))
        else:
            return r.name
    # sort rules for reproducibility
    rules = sorted(rules, key=key)
    tok_to_id = OrderedDict()#OrderedDict for reproducibility
    for i, r in enumerate(rules):
        tok_to_id[r] = i
    X, y = get_dataset(dataset)
    
    score_func = partial(mod.score, X=X, y=y, **params['score'])
    
    vocab_size = len(rules)
    emb_size = opt['emb_size']
    hidden_size = opt['hidden_size']
    nb_features = 2
    algo = opt['algo']
    gamma = opt['gamma']
    ih_std = opt['init_ih_std']
    hh_std = opt['init_hh_std']

    torch.manual_seed(random_state) # for LSTM initialization
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        nb_features=nb_features)
    
    def weights_init(m):
        if isinstance(m, nn.LSTM):
            m.weight_ih_l0.data.normal_(0, ih_std)
            m.weight_hh_l0.data.normal_(0, hh_std)
        elif isinstance(m, nn.Linear):
            xavier_uniform(m.weight.data)
            m.bias.data.fill_(0)
    model.apply(weights_init)
    
    algos = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}
    algo_cls = algos[algo['name']]
    algo_params = algo['params']
    optim = algo_cls(model.parameters(), **algo_params)
    rnn = RnnAdapter(
        model, 
        tok_to_id,
        random_state=random_state
    )
    wl = RnnWalker(
        grammar=grammar, 
        rnn=rnn,
        min_depth=min_depth, 
        max_depth=max_depth, 
        strict_depth_limit=strict_depth_limit,
    )
    codes, scores = rnn_optimize(
        score_func, 
        wl, 
        optim,
        nb_iter=nb_iter
    )
    stats = {'codes': codes, 'scores': scores}
    return stats



def plot(job_summary):
    db = load_db()
    job = db.get_job_by_summary(job_summary)
    scores = job['stats']['scores']
    #scores = np.max.accumulate(scores)
    #scores = pd.ewma(pd.Series(scores), span=1./0.01-1)
    fig = plt.figure()
    plt.plot(scores)
    plt.savefig('out.png')

def best_hypers(jobset='rnn_hypers_pipeline'):
    rng = np.random
    db = load_db()
    jobs = db.jobs_with(jobset=jobset, optimizer='rnn')
    crit = _auc
    jobs = sorted(jobs, key=lambda j:crit(j['stats']['scores']), reverse=True)
    for j in jobs:
        scores = j['stats']['scores']
        codes = j['stats']['codes']
        print(json.dumps(j['content'], indent=2), 
             j['summary'], 
             crit(scores), 
             np.max(scores),
             codes[np.argmax(scores)])

def test():
    rules = pipeline.rules
    tok_to_id = OrderedDict()#OrderedDict for reproducibility
    for i, r in enumerate(rules):
        tok_to_id[r] = i
    model = RnnModel(vocab_size=len(tok_to_id), nb_features=2) 
    rnn = RnnAdapter(model, tok_to_id)
    wl = RnnWalker(
        grammar=pipeline.grammar, 
        rnn=rnn,
        min_depth=1, 
        max_depth=5, 
        strict_depth_limit=False,
    )
    for _ in range(100):
        wl.walk()
        print(len(list(filter(lambda d:d.action=='rule', wl._decisions))))
 
if __name__ == '__main__':
    run([optim, plot, best_hypers, test])
