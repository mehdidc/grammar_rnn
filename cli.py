import warnings
import logging
from datetime import datetime
from collections import namedtuple

from clize import run

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

import torch

from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.random import RandomWalker
from grammaropt.types import Int, Float
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker

from grammar import generate_rules

from lightjob.cli import load_db
import random


warnings.filterwarnings("ignore")
fmt = ''
#logging.basicConfig(format=fmt)

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

hndl = logging.FileHandler('log',  mode='w')
hndl.setLevel('DEBUG')
log.addHandler(hndl)

hndl = logging.StreamHandler()
log.addHandler(hndl)

EPS = 1e-7


def as_str(terminals):
    return ''.join(map(str, terminals))


def evaluate(code, X, y, scoring=None, cv=5):
    try:
        clf = _build_estimator(code)
        scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)
    except Exception as ex:
        log.error('Error on code : {}'.format(code))
        log.error('Details : {}'.format(ex))
        log.error('')
        return 0.
    else:
        return float(np.mean(scores))


def _build_estimator(code):
    import sklearn
    import sklearn.kernel_approximation
    import sklearn.naive_bayes
    import sklearn.cluster
    import xgboost
    clf = eval(code)
    return clf

Config = namedtuple('Config',[
    'grammar',
    'nb_iter',
    'X',
    'y',
    'min_depth',
    'max_depth',
    'strict_depth_limit',
    'cv',
    'random_state'
])
def optim(*, optimizer=None, nb_iter=2, dataset='digits', 
          min_depth=1, max_depth=5, strict_depth_limit=False, 
          label='default', cv=5, random_state=None):
    start_time = str(datetime.now())
    optimizers = {'random': _optimize_random, 'rnn': _optimize_rnn}
    if not optimizer:
        optimizer = random.choice(list(optimizers.keys()))
    if not random_state:
        random_state = random.randint(1, 1000000)
    X, y = get_dataset(dataset)
    rules, types = generate_rules()
    grammar = build_grammar(rules, types=types)
    opt = optimizers[optimizer]
    config = Config(
        grammar=grammar,
        nb_iter=nb_iter,
        X=X, y=y,
        min_depth=min_depth, 
        max_depth=max_depth,
        strict_depth_limit=strict_depth_limit,
        cv=cv,
        random_state=random_state
    )
    stats = opt(config)
    end_time = str(datetime.now())
    log.info('Save in db')
    content = config._asdict()
    del content['X']
    del content['y']
    del content['grammar']
    db = load_db()
    s = db.add_job(content, stats=stats, label=label, dataset=dataset, optimizer=optimizer)
    db.job_update(s, {'start': start_time, 'end': end_time})


def _optimize_random(config):
    cf = config
    wl = RandomWalker(
        cf.grammar, 
        min_depth=cf.min_depth, max_depth=cf.max_depth, 
        strict_depth_limit=cf.strict_depth_limit, 
        random_state=cf.random_state)
    codes = []
    scores = []
    for it in range(cf.nb_iter):
        log.info('Generate code...')
        wl.walk()
        code = as_str(wl.terminals)
        log.info('Evaluate...')
        log.info(code)
        score = evaluate(code, cf.X, cf.y, cv=cf.cv)
        codes.append(code)
        scores.append(score)
        log.info('iteration {}'.format(it))
    return {'codes': codes, 'scores': scores}


def _optimize_rnn(config):
    cf = config
    rules = extract_rules_from_grammar(cf.grammar)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    # set hyper-parameters and build RNN model
    vocab_size = len(rules)
    emb_size = 128
    hidden_size = 128
    nb_features = 2
    lr = 1e-3
    gamma = 0.9
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        nb_features=nb_features)
    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    rnn = RnnAdapter(model, tok_to_id, random_state=cf.random_state)
    # optimization loop
    R_avg = 0.
    wl = RnnWalker(grammar=cf.grammar, rnn=rnn)
    codes = []
    scores = []
    for it in range(cf.nb_iter):
        log.info('Generate code...')
        wl.walk()
        code = as_str(wl.terminals)
        log.info('Evaluate...')
        R = evaluate(code, cf.X, cf.y, cv=cf.cv)
        R_avg = R_avg * gamma + R * (1 - gamma)
        model.zero_grad()
        log.info('Learn...')
        loss = (R - R_avg) * wl.compute_loss()
        loss.backward()
        optim.step()
        codes.append(code)
        scores.append(R)
    return {'codes': codes, 'scores': scores}


def get_dataset(name):
    from sklearn import datasets

    if name == 'digits':
        dataset = datasets.load_digits()
        X = dataset['data']
        y = dataset['target']
        return X, y
    elif name == "iris":
        dataset = datasets.load_iris()
        X = dataset['data']
        y = dataset['target']
        return X, y
    else:
        raise ValueError(name)


def show_grammar():
    rules, types = generate_rules()
    print(rules)


def show():
    rules, types = generate_rules()
    grammar = build_grammar(rules=rules, types=types)
    for _ in range(10):
        wl = RandomWalker(grammar, min_depth=1, max_depth=5, random_state=1)
        wl.walk()
        print(as_str(wl.terminals))

def plots(*, label='base'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    db = load_db()
    df = []
    for optim in ('rnn', 'random'):
        jobs = db.jobs_with(label=label, optimizer=optim)
        jobs = list(jobs)
        stats = [j['stats'] for j in jobs]
        for stat in stats:
            stat['optim'] = optim
            stat['score'] = max(stat['scores'])
            del stat['scores']
            df.append(stat)
    df = pd.DataFrame(df)
    print(df)
    df = df.groupby('optim').mean().reset_index()
    sns.barplot(x='optim', y='score', data=df)
    plt.savefig('out.png')


if __name__ == '__main__':
    run([optim, show_grammar, show, plots])
