import logging
from collections import OrderedDict
import numpy as np
from tpot.config_classifier import classifier_config_dict
from lightjob.utils import summarize

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import build_grammar
from grammaropt.types import Int
from grammaropt.types import Float
from grammaropt.random import RandomWalker

log = logging.getLogger(__name__)
hndl = logging.StreamHandler()
log.addHandler(hndl)


rules_tpl = r"""pipeline = "make_pipeline" op elements cm estimator cp
elements = (preprocessor cm elements) / preprocessor
preprocessor = {preprocessors} 
estimator = {estimators}
{body}
op = "("
cp = ")"
cm = ","
eq = "="
bool = "True" / "False"
none = "None"
"""

# for reproducibility, otherwise, without ordering, its lake we have a different random seeed
# each run
def _ordered(d):
    dout = OrderedDict()
    keys = sorted(d.keys())
    for k in keys:
        dout[k] = d[k]
    return dout
classifier_config_dict = _ordered(classifier_config_dict)


blacklist = [
    "tpot.built_in_operators.ZeroCount", 
    "sklearn.feature_selection.SelectFromModel", 
    "xgboost.XGBClassifier", 
    "sklearn.feature_selection.RFE"
]
def val_to_str(val):
    replace = dict(zip('0123456789', 'ijklmnopqr'))
    replace['.'] = 's'
    replace['e'] = 't'
    replace['-'] = 'u'
    s = str(val)
    s = [replace[c] for c in s]
    return ''.join(s)

def _generate_rules(d=classifier_config_dict, discrete=False):
    names = set(d.keys()) - set(blacklist)
    names = sorted(names)
    preprocessors = [k for k in names if 'Classifier' not in k and 'NB' not in k and 'svm' not in k]
    clf = list(set(names) - set(preprocessors))
    clf = sorted(clf)
    rules = OrderedDict()
    for e in preprocessors + clf:
        comps = ['"{}"'.format(e), "op"]
        d[e] = _ordered(d[e])
        for k, v in d[e].items():
            if type(v) == dict:
                # later
                continue
            ks = _slug(k)
            comps.append('"{}"'.format(k))
            comps.append("eq")
            comps.append(ks)
            comps.append("cm")
            if type(v) == list:
                if v == [True, False] or v == [True] or v == [False]:
                    rules[ks] = "bool"
                elif type(v[0]) == int:
                    if discrete:
                        for val in v:
                            rules[val_to_str(val)] = '"{}"'.format(val)
                        rules[ks] = " / ".join(map(val_to_str, v))
                    else:
                        rules[ks] = "int"
                elif type(v[0]) == float:
                    if discrete:
                        for val in v:
                            rules[val_to_str(val)] = '"{}"'.format(val)
                        rules[ks] = " / ".join(map(val_to_str, v))
                    else:
                        rules[ks] = "float"
                elif type(v[0]) == str:
                    rules[ks] = " / ".join('"\\"{}\\""'.format(val) for val in v)
                else:
                    raise ValueError(k, v)
            elif type(v) == range:
                if discrete:
                    for val in v:
                        rules[val_to_str(val)] = '"{}"'.format(val)
                    rules[ks] = " / ".join(map(val_to_str, v))
                else:
                    rules[ks] = "int"
            elif type(v) == np.ndarray:
                if 'int' in str(v.dtype):
                    if discrete:
                        for val in v:
                            rules[val_to_str(val)] = '"{}"'.format(val)
                        rules[ks] = " / ".join(map(val_to_str, v))
                    else:
                        rules[ks] = "int"     
                elif 'float' in str(v.dtype):
                    if discrete:
                        for val in v:
                            rules[val_to_str(val)] = '"{}"'.format(val)
                        rules[ks] = " / ".join(map(val_to_str, v))
                    else:
                        rules[ks] = "float"
                else:
                    raise ValueError(k, v)
            elif v == None:
                rules[ks] = "none"
            else:
                raise ValueError(ks, v)
        comps.append("cp")
        rules[_slug(e)] = ' '.join(comps)
    r  = ["{} = {}".format(k, v) for k, v in rules.items()]
    r = "\n".join(r)
    preprocessors = " / ".join(map(_slug, preprocessors))
    clf = " / ".join(map(_slug, clf))
    rules = rules_tpl.format(preprocessors=preprocessors, estimators=clf, body=r)
    types = OrderedDict()
    types["int"] = Int(2, 10)
    types["float"] = Float(0., 1.)
    return rules, types


def _slug(s):
    return s.lower().replace('.', '_')


def score(code, data, scoring=None, cv=5):
    X_train, X_test, y_train, y_test = data
    try:
        clf = _build_estimator(code)
        #clf.fit(X, y)
        clf.fit(X_train, y_train)
        score = (clf.predict(X_test) == y_test).mean()
        #scores = cross_val_score(clf, X, y, scoring=scoring, cv=StratifiedKFold(n_splits=cv, shuffle=True))
    except Exception as ex:
        log.error('Error on code : {}'.format(code))
        log.error('Details : {}'.format(ex))
        log.error('')
        return 0.
    else:
        print(score)
        return float(score)


def _build_estimator(code):
    clf = eval(code)
    return clf

discrete = True
rules, types = _generate_rules(discrete=discrete)
print(rules)
if discrete:
    grammar = build_grammar(rules)
else:
    grammar = build_grammar(rules, types)
rules = extract_rules_from_grammar(grammar)
