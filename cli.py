import os
import time
import json
import math
from functools import partial
from functools import wraps
from itertools import chain
from itertools import product
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
sns.set_style("white")


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
from hypers import datasets
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


def optim(jobset, *, db=None):
    

    params = generate_job(jobset=jobset)
    print(json.dumps(params, indent=2))
    dataset = params['dataset']
    optimizer = params['optimizer']['name']
    grammar = params['grammar']
    random_state = params['random_state']
    db_name = db 
    db = load_db(db_name)

    _assert_job(jobset, params, db=db)

    np.random.seed(random_state)

    _optim = optimizers[optimizer]
    start_time = datetime.now()
    stats = _optim(params)
    end_time = datetime.now()

    _assert_job(jobset, params, db=db)

    db.safe_add_job(
        params, 
        stats=stats, 
        jobset=jobset, 
        dataset=dataset, 
        grammar=grammar, 
        optimizer=optimizer,
        start=start_time,
        end=end_time)

def _assert_job(jobset, params, db):
    limit_per_optimizer = 10
    if jobset in ('pipeline', 'random_pipeline_for_prior'):
        jobs = db.jobs_with(jobset=jobset, optimizer=params['optimizer']['name'], dataset=params['dataset'])
        jobs = list(jobs)
        print(len(jobs))
        assert len(jobs) < limit_per_optimizer, (params['optimizer']['name'], params['dataset'], len(jobs))
        print('ok')


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
    nb_layers = opt['nb_layers']
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

    model = torch.load(
        filename, 
        map_location=lambda storage, loc: storage #  load on CPU
    )
    model.use_cuda = False
    
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


def _optim_finetune_rnn_from_params(params):
    dataset = params['dataset']
    random_state = params['random_state']

    torch.manual_seed(random_state) # for LSTM initialization

    opt = params['optimizer']['params']
    min_depth = opt['min_depth']
    max_depth = opt['max_depth']
    strict_depth_limit = opt['strict_depth_limit']
    nb_iter = opt['nb_iter']
    filename = opt['model']
    algo = opt['algo']
    gamma = opt['gamma']

    mod = grammars[params['grammar']]
    grammar = mod.grammar
    rules = mod.rules
    tok_to_id = _get_tok_to_id(rules)

    data = get_dataset(dataset)
    score_func = partial(mod.score, data=data, **params['score'])

    model = torch.load(
        filename, 
        map_location=lambda storage, loc: storage #  load on CPU
    )
    model.use_cuda = False

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
        nb_iter=nb_iter,
        gamma=gamma
    )
    stats = {'codes': codes, 'scores': scores}
    return stats


optimizers = {
    'rnn': _optim_rnn_from_params, 
    'random': _optim_random_from_params,
    'frozen_rnn': _optim_frozen_rnn_from_params,
    'prior_rnn': _optim_frozen_rnn_from_params,
    'finetune_rnn': _optim_finetune_rnn_from_params,
    'finetune_prior_rnn' : _optim_finetune_rnn_from_params
}



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


def learning_curve_plot(*, jobset=None, dataset=None, out='out.png', db=None, topk=1, optimizers=None):
    db = load_db(db)
    kw = {}
    if jobset:
        kw['jobset'] = jobset
    if dataset:
        kw['dataset'] = dataset
    jobs = db.jobs_with(**kw)

    rows = []
    for j in jobs:
        max_score = 0.
        scores = (j['stats']['scores'])
        codes = j['stats']['codes']
        code_to_score = {code:score for code, score in zip(codes, scores)}
        for it, score in enumerate(scores[topk-1:]):
            codes_so_far = list(set(codes[0:it+1]))
            if len(codes_so_far) < topk:
                continue
            codes_so_far = sorted(codes_so_far, key=lambda c:code_to_score[c], reverse=True)
            codes_so_far = codes_so_far[0:topk]
            top = [code_to_score[c] for c in codes_so_far]
            top_score = sum(top) / len(top) 
            rows.append({'score': top_score, 'optimizer': j['optimizer'], 'iter': it+topk, 'id': j['summary']})
    # learning curve with average performance
    df = pd.DataFrame(rows)
    fig = plt.figure()
    _plot_learning_curve(df, time='iter', score='score', optimizers=optimizers)
    plt.legend(loc='bottom left')
    plt.ylabel('accuracy(valid)')
    plt.savefig(out)
    plt.close(fig)

def rank_plot(*, jobset=None, dataset=None, out='out.png', db=None, optimizers=None):
    if dataset is None:
        df_list = []
        for d in datasets:
            df = _rank_df(jobset=jobset, dataset=d, db=db, optimizers=optimizers)
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df = df.groupby(('iter', 'optimizer')).mean().reset_index()
    else:
        df = _rank_df(jobset=jobset, dataset=dataset)
    fig = plt.figure(figsize=(10, 5))
    _plot_learning_curve(df, time='iter', score='rank', optimizers=optimizers)
    plt.legend(loc='best', bbox_to_anchor=(0.4, 0.7))
    plt.ylabel('rank')
    plt.savefig(out)
    plt.close(fig)

def _rank_df(jobset=None, dataset=None, db=None, optimizers=None):
    db = load_db(db)
    kw = {}
    if jobset:
        kw['jobset'] = jobset
    if dataset:
        kw['dataset'] = dataset
    jobs = db.jobs_with(**kw)
    jobs = list(jobs)
    #jobs = [j for j in jobs if j['optimizer'] in ('frozen_rnn', 'random', 'prior_rnn')]
    rows = []
    for j in jobs:
        if optimizers and j['optimizer'] not in optimizers:
            continue
        scores = (j['stats']['scores'])
        max_score = 0.
        for it, score in enumerate(scores):
            max_score = max(score, max_score)
            rows.append({'score': max_score, 'optimizer': j['optimizer'], 'iter': it, 'id': j['summary']})
    df = pd.DataFrame(rows)
    rows = []
    for it, group in df.groupby('iter'):
        d = group.groupby('optimizer').mean().reset_index()
        opts = d['optimizer'].values
        ranks = ((-d['score']).argsort() + 1).values
        for opt, rank in zip(opts, ranks):
            rows.append({'iter': it, 'optimizer': opt, 'rank': rank})
    df = pd.DataFrame(rows)
    return df

def time_to_reach_plot(jobset=None, dataset=None, out='out.png', db=None):
    values = np.linspace(0.5, 0.8, 10)
    #values = np.linspace(0, 0.6, 10)
    db = load_db(db)
    kw = {}
    if jobset:
        kw['jobset'] = jobset
    if dataset:
        kw['dataset'] = dataset
    jobs = db.jobs_with(**kw)
    rows = []
    for j in jobs:
        scores = (j['stats']['scores'])
        for value in values:
            ttr = _time_to_reach(scores, value)
            if np.isinf(ttr):
                ttr = len(scores)
            rows.append({'optimizer': j['optimizer'], 'ttr': ttr, 'value': value})
    df = pd.DataFrame(rows)
    fig = plt.figure()
    _plot_learning_curve(df, time='value', score='ttr')
    plt.legend()
    plt.savefig(out)
    plt.close(fig)


def stats(*, jobset=None, dataset=None, db=None):
    db = load_db(db)
    kw = {}
    if jobset:
        kw['jobset'] = jobset
    if dataset:
        kw['dataset'] = dataset
    jobs = db.jobs_with(**kw)
    rows = []
    for j in jobs:
        max_score = 0.
        scores = (j['stats']['scores'])
        codes = j['stats']['codes']
        for it, score in enumerate(scores):
            max_score = max(score, max_score)
            code = codes[it]
            rows.append({'score': score, 'optimizer': j['optimizer'], 'iter': it, 'id': j['summary'], 'dataset': j['dataset'], 'code': code})
    
    df = pd.DataFrame(rows)
    print('Total nb of (code, score) tuples : {}'.format(len(df)))
    print('List of Optimizers : ', df['optimizer'].unique())
    print('List of datasets : ', df['dataset'].unique())
    for opt, opt_grp in df.groupby('optimizer'):
        print(opt)
        for ds, ds_grp in opt_grp.groupby('dataset'):
            mu = ds_grp.groupby('iter').count()['id'].iloc[0]
            print('total eval : {:05d}, avg eval per iter : {:05d} Dataset : {} '.format(len(ds_grp), mu, ds))
    max_score = df['score'].max()
    print(df[df['score'] == max_score].values)
    df = df.groupby(['optimizer', 'id']).max().reset_index()
    df = df.groupby('optimizer').agg(('mean', 'std'), axis=1)
    print(df)


def test(*, jobset=None, dataset=None, db=None):
    db = load_db(db)
    kw = {}
    if jobset:
        kw['jobset'] = jobset
    if dataset:
        kw['dataset'] = dataset
    jobs = db.jobs_with(**kw)
    rows = []
    for j in jobs:
        scores = (j['stats']['scores'])
        codes = j['stats']['codes']
        for it, score in enumerate(scores):
            code = codes[it]
            rows.append({'score': score, 'optimizer': j['optimizer'], 'iter': it, 'id': j['summary'], 'dataset': j['dataset'], 'code': code})
    df = pd.DataFrame(rows)
    out = []
    for ds, ds_group in df.groupby('dataset'):
        print(ds)
        if ds == 'abalone':
            continue
        for opt, opt_group in ds_group.groupby('optimizer'):
            l = opt_group.sort_values(by='score', ascending=False).iloc[0]
            score = l['score']
            code = l['code']
            acc = _test_perf(code=code, dataset=ds)
            out.append({'test': acc, 'valid': score, 'dataset': ds, 'code': code, 'optimizer': opt})
            print(acc, score, opt)
    df = pd.DataFrame(out)
    df.to_csv('plots/test.csv')


def _plot_learning_curve(df, time='iter', score='score', optimizers=None):
    if not optimizers:
        optimizers = (
            'random', 
            'frozen_rnn', 
            'prior_rnn', 
            'finetune_prior_rnn', 
            'finetune_rnn'
        )
    for opt in optimizers:
        color = {
            'rnn': 'blue', 
            'random': 'green', 
            'frozen_rnn': 'orange', 
            'finetune_rnn': 'purple', 
            'prior_rnn': 'red', 
            'finetune_prior_rnn': 'blue'}[opt]
        rename = {
            'frozen_rnn': 'meta-rnn', 
            'prior_rnn': 'prior-rnn', 
            'finetune_rnn': 'finetuned-meta-rnn',
            'finetune_prior_rnn': 'finetuned-prior-rnn',
            'random': 'random'
        }

        d = df[df['optimizer'] == opt]
        if len(d) == 0:
            continue
        #d = d.groupby(time).agg(['mean', 'std']).reset_index()
        d = d.groupby(time).agg(['mean',  'std']).reset_index()
        d = d.sort_values(by=time)
        mu, std = d[score]['mean'], d[score]['std']
        #v = mu - std if not np.any(np.isnan(std)) else mu
        plt.plot(d[time], mu, label=rename[opt], color=color)
        #plt.errorbar(x=d[time], y=mu, yerr=std, color=color, label=rename[opt)
        plt.fill_between(d[time], mu - std, mu + std, alpha=0.1, color=color, linewidth=0)
        plt.xlabel('nb. of evaluations')


def _bs_mean(x):
    random_state = hash(tuple(x.tolist())) % 4294967295
    rng = np.random.RandomState(random_state)
    bs = rng.randint(0, len(x), size=10000)
    return x.values[bs].mean()
_bs_mean.__name__ = 'mean'

def _bs_std(x):
    random_state = hash(tuple(x.tolist())) % 4294967295
    rng = np.random.RandomState(random_state)
    bs = rng.randint(0, len(x), size=10000)
    return x.values[bs].std()
_bs_std.__name__ = 'std'

def fit(*, jobset='pipeline', grammar='pipeline', out_folder='models', exclude_dataset=None, include_dataset=None, cuda=False, resample=False, normalize_loss=False, nb_epochs=8, db=None):
    mod = grammars[grammar]
    grammar = mod.grammar
    rules = mod.rules
    gamma = 0.99
    min_depth = 1
    max_depth = 5
    strict_depth_limit = False
    lr = 1e-4
    
    hidden_size = 128
    emb_size = 128
    num_layers = 2
    nb_features = 2
    ih_std = 0.08
    hh_std = 0.08
    
    out_filename = os.path.join(out_folder, 'model.th')
    assert not os.path.exists(out_filename)
    model = RnnModel(
        vocab_size=len(rules),
        hidden_size=hidden_size,
        emb_size=emb_size,
        num_layers=num_layers,
        nb_features=nb_features,
        use_cuda=cuda,
    )
    model.apply(partial(_weights_init, ih_std=ih_std, hh_std=hh_std))
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
    db = load_db(db)
    jobs = db.jobs_with(jobset=jobset, optimizer='random')
    jobs = list(jobs)
    
    assert (include_dataset and not exclude_dataset) or (not include_dataset and exclude_dataset)
    if include_dataset:
        print(len(jobs))
        jobs = [j for j in jobs if j['dataset'] == include_dataset]
        print(len(jobs))
    elif exclude_dataset:
        print(len(jobs))
        jobs = [j for j in jobs if j['dataset'] != exclude_dataset]
        print(len(jobs))
    assert len(jobs)
    X, Y = _build_dataset_from_jobs(jobs)
    np.savez(os.path.join(out_folder, 'src.npz'), X=X, Y=Y)
    avg_loss = 0.
    nb_updates = 0

    X = np.array(X)
    Y = np.array(Y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    if resample:
        X, Y = _resample(X, Y, nb=4)
    losses = []
    t0 = time.time()
    print(X.shape)
    for epoch in range(nb_epochs):
        t0 = time.time()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        for i, (x, y) in enumerate(zip(X, Y)):
            dwl = RnnDeterministicWalker.from_str(grammar, rnn, x)
            model.zero_grad()
            dwl.walk()
            loss = float(y) * dwl.compute_loss()
            if normalize_loss:
                loss /=  len(dwl.decisions)
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), 2)
            optim.step()
            avg_loss = gamma * avg_loss + (loss.data[0]) * (1 - gamma)
            losses.append(loss.data[0])
            if nb_updates % 100 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print('Epoch {:04d}/{:04d} Example {:04d}/{:04d} Avg loss : {:.3f} time {:.3f}'.format(epoch + 1, nb_epochs, i + 1, len(X), avg_loss, dt))
                pd.Series(losses).to_csv(os.path.join(out_folder, 'stats.csv'))
            nb_updates += 1
        delta_t = time.time() - t0
        print('Time elapsed in epoch {:04f} : {:.3f}s'.format(epoch + 1, delta_t))
        torch.save(model, out_filename)
    

def _resample(X, Y, nb=1):
    p = np.array(Y)
    p /= p.sum()
    X = np.random.choice(X, size=nb * len(X), p=p, replace=True)
    Y = np.ones(len(X))
    return X, Y


def _build_dataset_from_jobs(jobs):
    code_to_score = defaultdict(list)
    for j in jobs:
        for code, score in zip(j['stats']['codes'], j['stats']['scores']):
            code_to_score[code].append(score)
    X = []
    y = []
    for code, scores in code_to_score.items():
        X.append(code)
        y.append(float(np.mean(scores)))
    return X, y

def clean():
    #out_pipeline -> pipeline
    db = load_db()

    #jobs = db.jobs_with(jobset='out_pipeline')
    #for j in jobs:
    #    db.job_update(j['summary'], {'jobset': 'pipeline'})

    #jobs = db.jobs_with(jobset='random_pipeline')
    #for j in jobs:
    #    db.job_update(j['summary'], {'jobset': 'pipeline'})
    
    #jobs = db.jobs_with(jobset='rnn_pipeline')
    #for j in jobs:
    #    db.job_update(j['summary'], {'jobset': 'pipeline'})
    for ds, opt in product(datasets, optimizers.keys()):
        jobs = db.jobs_with(dataset=ds, optimizer=opt, jobset='pipeline')
        jobs = list(jobs)
        if len(jobs) <= 10:
            continue
        jobs = sorted(jobs, key=lambda j:pd.to_datetime(j['end']))
        print(len(jobs[10:]), len(jobs[0:10]), len(jobs))
        for j in jobs[10:]:
            db.delete({'summary': j['summary']})

def plots(*, db=None):
    test_plot(out='plots/test.png')
    opt = ('prior_rnn', 'frozen_rnn', 'random')
    for dataset in datasets:
        print('{}...'.format(dataset))
        
        out = 'plots/learning_curve/{}.png'.format(dataset)
        learning_curve_plot(jobset='pipeline', dataset=dataset, out=out, db=db, optimizers=opt)

        out = 'plots/learning_curve_prior/{}.png'.format(dataset)
        learning_curve_plot(jobset='pipeline', dataset=dataset, out=out, db=db, optimizers=('prior_rnn', 'finetune_prior_rnn'))

        out = 'plots/learning_curve_meta/{}.png'.format(dataset)
        learning_curve_plot(jobset='pipeline', dataset=dataset, out=out, db=db, optimizers=('frozen_rnn', 'finetune_rnn'))

        out = 'plots/learning_curve_top20/{}.png'.format(dataset)
        learning_curve_plot(jobset='pipeline', dataset=dataset, out=out, db=db,  topk=20)

        out = 'plots/learning_curve_top10/{}.png'.format(dataset)
        learning_curve_plot(jobset='pipeline', dataset=dataset, out=out, db=db,  topk=10)

        out = 'plots/time_to_reach/{}.png'.format(dataset)
        time_to_reach_plot(jobset='pipeline', dataset=dataset, out=out, db=db)

        out = 'plots/rank/{}.png'.format(dataset)
        rank_plot(jobset='pipeline', dataset=dataset, out=out, db=db, optimizers=('prior_rnn', 'frozen_rnn', 'random'))

    out = 'plots/ranks_all.png'
    rank_plot(jobset='pipeline', out=out, db=db)

    out = 'plots/ranks.png'
    rank_plot(jobset='pipeline', out=out, db=db, optimizers=opt)



def _test_perf(*, code='make_pipeline(sklearn.linear_model.LogisticRegression())', dataset='car'):
    X_train, X_valid, y_train, y_valid = get_dataset(dataset, which='train')
    X = np.concatenate((X_train, X_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)
    #X = X_train
    #y = y_train
    np.random.seed(42)
    clf = eval(code)
    clf.fit(X, y)
    X_test, y_test = get_dataset(dataset, which='test')
    return ((clf.predict(X_test) == y_test).mean())


def test_plot(out='test.png'):
    import seaborn as sns
    df = pd.read_csv('plots/test.csv')
    #df = df[ (df['optimizer'] == 'prior_rnn') | (df['optimizer']=='random')| (df['optimizer']=='frozen_rnn')]
    rename = {
        'frozen_rnn': 'meta-rnn', 
        'prior_rnn': 'prior-rnn', 
        'finetune_rnn': 'finetuned-meta-rnn',
        'finetune_prior_rnn': 'finetuned-prior-rnn',
        'random': 'random'
    }
    df['optimizer'] = df['optimizer'].apply(lambda name:rename[name])

    for ds, gr_ds in df.groupby('dataset'):
        best = gr_ds[['optimizer', 'test', 'code']].sort_values(by='test', ascending=False).iloc[0]
        print('{}#dataset : {}, found by {}'.format(best['code'], ds, best['optimizer']))


    for ds, gr_ds in df.groupby('dataset'):
        print(ds)
        gr_ds['err'] = 1.0 - gr_ds['test']
        print(gr_ds[['optimizer', 'test']])

    palette = {
        'meta-rnn': 'orange', 
        'prior-rnn': 'red', 
        'random': 'green',
        'finetuned-meta-rnn': 'blue',
        'finetuned-prior-rnn': 'purple'
    }
    order = ['meta-rnn', 'finetuned-meta-rnn', 'prior-rnn', 'finetuned-prior-rnn', 'random']
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='dataset', y='test', hue='optimizer', data=df, palette=palette, hue_order=order)
    plt.xlabel('dataset')
    plt.ylabel('accuracy(test)')
    plt.legend(loc='best')
    plt.savefig(out)
    plt.close(fig)


def new():
    db = load_db()
    jobs = db.jobs_with(jobset='pipeline', optimizer='random')
    jobs = list(jobs)
    random.shuffle(jobs)
    rnd = []
    rnd_prior = []
    N = 10
    D = 10
    for ds in datasets:
        ds_only = [j for j in jobs if j['dataset'] == ds]
        rnd.extend(ds_only[0:N])
        rnd_prior.extend(ds_only[N:2*N])
    assert len(rnd) == N * D, len(rnd)
    assert len(rnd_prior) == N * D, len(rnd_prior)
    db  = load_db('.new')
    for j in rnd:
        meta = j.copy()
        del meta['content']
        db.safe_add_job(j['content'], **meta)

    for j in rnd_prior:
        meta = j.copy()
        del meta['content']
        meta['jobset'] = 'random_pipeline_for_prior'
        db.safe_add_job(j['content'], **meta)

def info():
    for ds in datasets:
        X_train, X_valid, y_train, y_valid = get_dataset(ds, which='train')
        n_test, y_test = get_dataset(ds, which='test')
        print('{} & {} & {} & {} & {} & {} \\\\ \\hline'.format(ds, X_train.shape[0], X_valid.shape[0], X_test.shape[0], X_train.shape[1], len(set(y_train))))

if __name__ == '__main__':
    run([optim, plot, best_hypers, learning_curve_plot, time_to_reach_plot, fit, clean, plots, rank_plot, stats, new, info, test, test_plot])
