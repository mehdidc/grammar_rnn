from collections import OrderedDict
import numpy as np
from tpot.config_classifier import classifier_config_dict

from grammaropt.grammar import build_grammar
from grammaropt.types import Int
from grammaropt.types import Float
from grammaropt.random import RandomWalker
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


def generate_rules(d=classifier_config_dict):
    names = set(d.keys()) - set(["tpot.built_in_operators.ZeroCount", "sklearn.feature_selection.SelectFromModel", "xgboost.XGBClassifier"])
    names = sorted(names)
    preprocessors = [k for k in names if 'Classifier' not in k and 'NB' not in k and 'svm' not in k]
    clf = list(set(names) - set(preprocessors))
    clf = sorted(clf)
    #preprocessors = preprocessors[0:2]
    #clf = clf[0:2]
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
                    rules[ks] = "int"
                elif type(v[0]) == float:
                    rules[ks] = "float"
                elif type(v[0]) == str:
                    rules[ks] = " / ".join('"\\"{}\\""'.format(val) for val in v)
                else:
                    raise ValueError(k, v)
            elif type(v) == range:
                rules[ks] = "int"
            elif type(v) == np.ndarray:
                if 'int' in str(v.dtype):
                    rules[ks] = "int"
                elif 'float' in str(v.dtype):
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
