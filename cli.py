import json
import math
from functools import partial
from functools import wraps
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

from data import get_dataset
import pipeline
import formula
from hypers import generate_job
from hypers import _auc
from hypers import _monotonicity
from hypers import _corr
from hypers import _time_to_reach

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
hndl = logging.StreamHandler()
log.addHandler(hndl)


grammars = {
    'pipeline': pipeline, 
    'formula': formula
}


def optim(jobset):
    
    optimizers = {
        'rnn': _optim_rnn_from_params, 
        'random': _optim_random_from_params,
        'frozen_rnn': _optim_frozen_rnn_from_params
    }

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

    data = get_dataset(dataset)
    
    score_func = partial(mod.score, data=data, **params['score'])

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

    tok_to_id = _get_tok_to_id(rules)
    data = get_dataset(dataset)    
    score_func = partial(mod.score, data=data, **params['score'])
    
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
    model.apply(partial(_weights_init, ih_std=ih_std, hh_std=hh_std))
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
    codes, scores = _rnn_optimize(
        score_func, 
        wl, 
        optim,
        nb_iter=nb_iter,
        gamma=gamma,
    )
    stats = {'codes': codes, 'scores': scores}
    return stats


def _get_tok_to_id(rules):
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
    return tok_to_id


def _weights_init(m, ih_std=0.08, hh_std=0.08):
    if isinstance(m, nn.LSTM):
        m.weight_ih_l0.data.normal_(0, ih_std)
        m.weight_hh_l0.data.normal_(0, hh_std)
    elif isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


def _rnn_optimize(func, walker, optim, nb_iter=10, gamma=0.9):
    wl = walker
    model = wl.rnn.model
    X = []
    y = []
    R_avg = 0.
    for it in range(nb_iter):
        wl.walk()
        code = as_str(wl.terminals)
        R = func(code)
        R_avg = R_avg * gamma + R * (1 - gamma)
        model.zero_grad()
        loss = (R - R_avg) * wl.compute_loss()
        loss.backward()
        optim.step()
        X.append(code)
        y.append(R)
    return X, y


def _rnn_optimize_greedy_epsilon(func, walker, optim, nb_iter=10, gamma=0.9, decay=float('inf')):
    wl = walker
    model = wl.rnn.model
    randomwl = RandomWalker(
        grammar=wl.grammar, 
        min_depth=wl.min_depth, 
        max_depth=wl.max_depth, 
        strict_depth_limit=wl.strict_depth_limit,
    )
    rng = wl.rnn.rng
    randomwl.rng = rng
    X = []
    y = []
    R_avg = 0.
    for it in range(nb_iter):
        eps = 1. - 1. / (1. + decay * it)
        if rng.uniform() <= eps:
            wl.walk()
            code = as_str(wl.terminals)
        else:
            randomwl.walk()
            code = as_str(randomwl.terminals)
        R = func(code)
        R_avg = R_avg * gamma + R * (1 - gamma)

        model.zero_grad()
        dwl = RnnDeterministicWalker.from_str(wl.grammar, wl.rnn, code)
        dwl.walk()
        loss = (R - R_avg) * dwl.compute_loss()
        loss.backward()
        optim.step()
        X.append(code)
        y.append(R)
    return X, y


def _optim_frozen_rnn_from_params(params):
    dataset = params['dataset']
    random_state = params['random_state']

    torch.manual_seed(random_state) # for LSTM initialization

    opt = params['optimizer']['params']
    min_depth = opt['min_depth']
    max_depth = opt['max_depth']
    strict_depth_limit = opt['strict_depth_limit']
    nb_iter = opt['nb_iter']
    filename = opt['model']

    mod = grammars[params['grammar']]
    grammar = mod.grammar
    rules = mod.rules
    tok_to_id = _get_tok_to_id(rules)

    data = get_dataset(dataset)
    score_func = partial(mod.score, data=data, **params['score'])

    model = torch.load(filename)
    
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
    codes, scores = random_optimize(
        score_func, 
        wl, 
        nb_iter=nb_iter
    )
    stats = {'codes': codes, 'scores': scores}
    return stats


def best_hypers(jobset='rnn_hypers_pipeline'):
    rng = np.random
    db = load_db()
    jobs = db.jobs_with(jobset=jobset)
    #crit = wraps(_time_to_reach)(partial(_time_to_reach, val=0.5))
    crit = max
    jobs = sorted(jobs, key=lambda j:crit(j['stats']['scores']), reverse=True)
    rows = []
    for j in jobs:
        scores = j['stats']['scores']
        codes = j['stats']['codes']
        val = crit(scores)
        if np.isinf(val):
            val = len(scores)
        rows.append({'optimizer': j['optimizer'], 'val': val})
        print('{} {} {}:{:.5f} max:{:.5f}, best:{} best found at:{}'.format(
             json.dumps(j['content'], indent=2), 
             j['summary'], 
             crit.__name__,
             val, 
             np.max(scores),
             codes[np.argmax(scores)],
             np.argmax(scores)))

    df = pd.DataFrame(rows)
    print(df.groupby('optimizer').agg(['mean', 'std']))

def plot(job_summary):
    db = load_db()
    job = db.get_job_by_summary(job_summary)
    scores = job['stats']['scores']
    #scores = np.max.accumulate(scores)
    #scores = pd.ewma(pd.Series(scores), span=1./0.01-1)
    fig = plt.figure()
    plt.plot(scores)
    plt.savefig('out.png')


def learning_curve_plot(jobset='pipeline'):
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
    # learning curve with average performance
    df = pd.DataFrame(rows)
    _plot_learning_curve(df, time='iter', score='score')
    plt.legend()
    plt.savefig('out.png')


def time_to_reach_plot(jobset='pipeline'):
    values = np.linspace(0.6, 0.7, 10)
    #values = np.linspace(0, 0.6, 10)
    db = load_db()
    jobs = db.jobs_with(jobset=jobset)
    rows = []
    for j in jobs:
        scores = (j['stats']['scores'])
        for value in values:
            ttr = _time_to_reach(scores, value)
            if np.isinf(ttr):
                ttr = len(scores)
            rows.append({'optimizer': j['optimizer'], 'ttr': ttr, 'value': value})
    df = pd.DataFrame(rows)
    _plot_learning_curve(df, time='value', score='ttr')
    plt.legend()
    plt.savefig('out.png')


def _plot_learning_curve(df, time='iter', score='score'):
    for opt in ('rnn', 'random'):
        color = {'rnn': 'blue', 'random': 'green'}[opt]
        d = df[df['optimizer'] == opt]
        d = d.groupby(time).agg(['mean', 'std']).reset_index()
        d = d.sort_values(by=time)
        mu, std = d[score]['mean'], d[score]['std']
        plt.plot(d[time], mu, label=opt, color=color)
        plt.fill_between(d[time], mu - std, mu + std, alpha=0.2, color=color, linewidth=0)


def test():
    from queue import PriorityQueue
    from heapq import heappush, heappop
    mod = pipeline
    dataset = 'redwine'
    gamma = 0.9
    batch_size = 10
    qsize = 100
    rules = mod.rules
    tok_to_id = {r: i for i, r in enumerate(rules)}
    model = RnnModel(vocab_size=len(tok_to_id), nb_features=2, hidden_size=32) 
    model.apply(_weights_init)
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    rnnwl = RnnWalker(
        grammar=mod.grammar, 
        rnn=rnn,
        min_depth=1, 
        max_depth=5, 
        strict_depth_limit=False,
    )
    randomwl = RandomWalker(
        grammar=mod.grammar,
        min_depth=1,
        max_depth=5,
        strict_depth_limit=False,
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.0006)
    X = []
    y = []
    q = []
    R_avg = 0.
    R_max = 0.
    avg_loss = 0.
    X, y = get_dataset(dataset)
    rng = np.random
    R_list = []
    code_list = []
    eps = 0.8
    alpha_avg = 0.
    for it in range(1000):
        # generate
        #eps = (alpha_avg + 1.0) / 2.0
        eps = 1.0
        if rng.uniform() <= eps:
            rnnwl.walk()
            code = as_str(rnnwl.terminals)
        else:
            randomwl.walk()
            code = as_str(randomwl.terminals)
        # get score
        R = mod.score(code, X, y)
        R_max = max(R, R_max)
        R_avg = R_avg * gamma + R * (1 - gamma)
        code_list.append(code)
        R_list.append(R)
        # update
        wl = RnnDeterministicWalker.from_str(mod.grammar, rnn, code)
        wl.walk()
        model.zero_grad()
        alpha = (R - R_avg)
        alpha = R
        alpha_avg = alpha_avg * 0.9 + alpha * 0.1
        loss = alpha * wl.compute_loss()
        avg_loss = avg_loss * 0.9 + loss.data[0] * 0.1 
        loss.backward()
        optim.step()

        print('R:{:.3f} alpha_avg:{:.3f} eps:{:.3f} code:{}'.format(R, alpha_avg, eps, code))
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
    print(max(R_list))
    fig = plt.figure()
    plt.plot(R_list)
    plt.savefig('out.png')


def fit(*, jobset='pipeline', grammar='pipeline', out='model.th', cuda=False):
    nb_epochs = 1
    mod = grammars[grammar]
    grammar = mod.grammar
    rules = mod.rules
    gamma = 0.9
    min_depth = 1
    max_depth = 5
    strict_depth_limit = False

    lr = 1e-3
    model = RnnModel(
        vocab_size=len(rules),
        hidden_size=256,
        emb_size=128,
        cuda=cuda
    )
    model.apply(partial(_weights_init, ih_std=0.08, hh_std=0.08))
    if cuda:
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    tok_to_id = _get_tok_to_id(rules)
    rnn = RnnAdapter(model, tok_to_id)
    rnnwl = RnnWalker(
        grammar=grammar, 
        rnn=rnn, 
        min_depth=min_depth,
        max_depth=max_depth, 
        strict_depth_limit=strict_depth_limit
    )

    db = load_db()
    jobs = db.jobs_with(jobset=jobset)
    jobs = list(jobs)
    X, Y = _build_dataset_from_jobs(jobs)
    print(len(X), len(Y))
    avg_loss = 0.
    nb_updates = 0
    X = X[0:1]
    Y = Y[0:1]
    for epoch in range(nb_epochs):
        print('Epoch {}'.format(epoch))
        
        for x, y in zip(X, Y):
            print(x)
            dwl = RnnDeterministicWalker.from_str(grammar, rnn, x)
            dwl.walk()
            loss = y * dwl.compute_loss()
            loss.backward()
            optim.step()
            avg_loss = gamma * avg_loss + loss.data[0] * (1 - gamma)
            nb_updates += 1
            if nb_updates % 100 == 0:
                print('Avg loss : {.3f}'.format(avg_loss))
                rnnwl.walk()
                expr = as_str(wl.terminals)
                print(expr)
     

def _build_dataset_from_jobs(jobs):
    code_to_score = defaultdict(list)
    for j in jobs:
        for code, score in zip(j['stats']['codes'], j['stats']['scores']):
            code_to_score[code].append(score)
    X = []
    y = []
    for code, scores in code_to_score.items():
        X.append(code)
        y.append(np.mean(scores))
    return X, y

if __name__ == '__main__':
    run([optim, plot, best_hypers, test, learning_curve_plot, time_to_reach_plot, fit])
