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
from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker
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
from hypers import _auc, _monotonicity, _corr

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


def best_hypers(jobset='rnn_hypers_pipeline'):
    rng = np.random
    db = load_db()
    jobs = db.jobs_with(jobset=jobset, optimizer='rnn')
    crit = _corr
    jobs = sorted(jobs, key=lambda j:crit(j['stats']['scores']), reverse=True)
    for j in jobs:
        scores = j['stats']['scores']
        codes = j['stats']['codes']
        print(json.dumps(j['content'], indent=2), 
             j['summary'], 
             crit(scores), 
             np.max(scores),
             codes[np.argmax(scores)])


def plot(job_summary):
    db = load_db()
    job = db.get_job_by_summary(job_summary)
    scores = job['stats']['scores']
    #scores = np.max.accumulate(scores)
    #scores = pd.ewma(pd.Series(scores), span=1./0.01-1)
    fig = plt.figure()
    plt.plot(scores)
    plt.savefig('out.png')


def plots(jobset='pipeline'):
    db = load_db()
    jobs = db.jobs_with(jobset=jobset)
    rows = []
    for j in jobs:
        max_score = 0.
        scores = (j['stats']['scores'])
        for it, score in enumerate(scores):
            max_score = max(score, max_score)
            #top = sorted(scores[0:it + 1])[::-1][0:20]
            #top_score = sum(top) / len(top) 
            rows.append({'score': max_score, 'optimizer': j['optimizer'], 'iter': it, 'id': j['summary']})
    df = pd.DataFrame(rows)
    for opt in ('rnn', 'random'):
        color = {'rnn': 'blue', 'random': 'green'}[opt]
        d = df[df['optimizer'] == opt]
        d = d.groupby('iter').agg(['mean', 'std']).reset_index()
        d = d.sort_values(by='iter')
        mu, std = d['score']['mean'], d['score']['std']
        plt.plot(d['iter'], mu, label=opt, color=color)
        plt.fill_between(d['iter'], mu - std, mu + std, alpha=0.2, color=color, linewidth=0)
    plt.legend()
    plt.savefig('out.png')
    """
    print(df.groupby(['optimizer', 'id']).max().reset_index().groupby('optimizer').agg(('mean', 'std'))['score'])
    plt.clf()
    for opt in ('rnn', 'random'):
        color = {'rnn': 'blue', 'random': 'green'}[opt]
        d = df[df['optimizer'] == opt]
        d = d.sort_values(by='score', ascending=False)
        id_ = d.iloc[0]['id']
        d = df[df['id']==id_]
        d = d.sort_values(by='iter')
        plt.plot(d['iter'], d['score'], label=opt, color=color)
    plt.legend()
    plt.savefig('out.png')
    """


def test():
    from queue import PriorityQueue
    from heapq import heappush, heappop
    mod = formula
    dataset = 'x*x+cos(x)*sin(x)'
    gamma = 0.5
    batch_size = 10
    qsize = 100
    q = []
    rules = mod.rules
    tok_to_id = {r: i for i, r in enumerate(rules)}
    model = RnnModel(vocab_size=len(tok_to_id), nb_features=2, hidden_size=256) 
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    rnnwl = RnnWalker(
        grammar=mod.grammar, 
        rnn=rnn,
        min_depth=1, 
        max_depth=10, 
        strict_depth_limit=False,
    )
    randomwl = RandomWalker(
        grammar=mod.grammar,
        min_depth=1,
        max_depth=10,
        strict_depth_limit=False
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = []
    y = []
    R_avg = 0.
    X, y = get_dataset(dataset)
    rng = np.random

    R_list = []
    code_list = []
    eps = 0.8
    for it in range(1000):
        # generate
        if rng.uniform() <= eps:
            rnnwl.walk()
            code = as_str(rnnwl.terminals)
        else:
            print('random')
            randomwl.walk()
            code = as_str(randomwl.terminals)
        # get score
        R = mod.score(code, X, y)
        R_avg = R_avg * gamma + R * (1 - gamma)
        code_list.append(code)
        R_list.append(R)
        l = list(filter(lambda r:r>0, R_list))
        print(R, code, _corr(l))
        # update
        wl = RnnDeterministicWalker.from_str(mod.grammar, rnn, code)
        wl.walk()
        model.zero_grad()
        loss = (R - R_avg) * wl.compute_loss()
        #pred = model.out_value(rnnwl._state[0].view(1, -1))
        #loss_score = (pred[0, 0] - R) ** 2
        #loss += loss_score
        loss.backward()
        optim.step()

        """ 
        val = (R, code)
        if len(q) < qsize:
            heappush(q, val)
        else:
            R_prev, code_prev = heappop(q)
            if R_prev > R:
                R = R_prev
                code = code_prev
                val = (R, code)
            heappush(q, val)
        if it < batch_size:
            continue
        loss = 0.
        model.zero_grad()
        q_R_list = [mr for mr, _ in q]
        q_code_list = [code for _, code in q]
        p = np.array(q_R_list)
        p /= p.sum()
        p[-1] = 1 - p[0:-1].sum()
        for _ in range(batch_size):
            idx = rng.choice(np.arange(len(q_R_list)))
            code = q_code_list[idx]
            R = q_R_list[idx]
            wl = RnnDeterministicWalker.from_str(mod.grammar, rnn, code)
            wl.walk()
            loss += (R - R_avg) * wl.compute_loss()
            pred = model.out_value(wl._state[0].view(1, -1))
            loss_score = (pred[0, 0] - R) ** 2
            loss += loss_score
     
        loss /= batch_size
        loss.backward()
        optim.step()
        """
    fig = plt.figure()
    plt.plot(R_list)
    plt.savefig('out.png')
 
if __name__ == '__main__':
    run([optim, plot, best_hypers, test, plots])
