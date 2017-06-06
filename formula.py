import numpy as np
from numpy import exp, cos, sin
from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.types import Int

rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / "x" / int
    int =  "1" / "2" / "3" / "4" / "5" / "6" / "7" / "8" / "9"
    po = "("
    pc = ")"
"""
grammar = build_grammar(rules)
rules = extract_rules_from_grammar(grammar)

def score(code, data, thresh=0.1):
    X, y = data
    y_pred = _eval(code, X)
    return _score_formula(y, y_pred, thresh=thresh)

def _eval(code, x):
    return eval(code)

def _score_formula(y_true, y_pred, thresh=0.1):
    score = (np.abs(y_true - y_pred) <= thresh).mean()
    score = float(score)
    return score
