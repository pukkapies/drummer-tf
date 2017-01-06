"""
@author: Samer
"""
from functools import partial, reduce

class Fail(Exception):
    pass

def head(l):           return l[0]
def tail(l):           return l[1:]

# def fst((x, _)):  return x
# def snd((_, x)):  return x
# def swap((x, y)): return (y, x)
def identity(x):  return x
def const(x):     return lambda _: x
def flip(f):      return lambda x, y: f(y, x)
def app(f, x): return f(x)
def apply_to(x):  return lambda f: f(x)
def pair_with(x): return lambda y: (x, y)
def divby(k):     return lambda x: x/k

def guard(f, x):
    if f(x):
        return x
    raise Fail

def exclude_none(x, msg=""):
    """ :: maybe(A), msg:str='' -> A
    Throw exception with msg if x is None, otherwise return x
    """
    if x is None:
        raise LookupError(msg)
    return x

def if_none_else(fallback, value):
    """ :: A, maybe(A) -> A """
    return fallback if value is None else value

def member(X):    return lambda x: x in X
def not_member(X): return lambda x: x not in X

def composeAll(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) # == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    if len(args)==0:
        return identity
    return partial(reduce, compose)(*args)

def compose2(f, g): return lambda x: f(g(x))
def compose(*args): return reduce(compose2, args)

# def ffst(f): return lambda (x, y): (f(x), y)
# def fsnd(f): return lambda (x, y): (x, f(y))

def bind(f, *args, **kwargs): return partial(f, *args, **kwargs)

def split(n, l):
    return l[:n], l[n:]

def foldl(f, xs, s): return reduce(flip(f), xs, s)
def foldr(f, xs, s): return reduce(flip(f), reversed(xs), s)

def map_accum(f, xs, s):
    """ :: (a,s->(b,s)), list(a), s -> (list(b), s) """
    ys=[]
    for x in xs:
        (y, s) = f(x, s)
        ys.append(y)
    return (ys, s)

def map_filter(f, items):
    return [x[0] for x in map(f, items) if x != []]

def unsingleton(l):
    if len(l) != 1:
        raise ValueError(("List is not a singleton", l))
    return head(l)

def diff_with(f, xs):
    return [f(xs[i], xs[i-1]) for i in range(1, len(xs))]

def scanr(f, xs, s):
    ss=[]
    for x in reversed(xs):
        s = f(x, s)
        ss.insert(0, s)
    return ss

def partition(cond, items):
    """ :: (a->bool), list(a) -> (list(a), list(a)) """
    (sheep, goats) = ([], [])
    for i in items: (sheep if cond(i) else goats).append(i)
    return (goats, sheep)

def maybe(f, x):
    """ :: (A->B), maybe(A) -> maybe(B)
    Apply function if not None, otherwise, just keep the None."""
    if x is None:
        return None
    else:
        return f(x)

def table(domain, f):
    return {x: f(x) for x in domain}.__getitem__

def memoise(f):
    memo = {}
    def g(x):
        y = memo.get(x, None)
        if y is None:
            y = f(x)
            memo[x] = y
        return y
    return g

def lazy(f):
    result = [None]
    def g():
        if result[0] is None:
            result[0] = f()
        return result[0]
    return g

def unzip(tuples):
    """
    :: list((a,b,...)) -> list(a), list(b), ... | ()
    Unzip list of tuples into a tuple of lists. Works for any arity.
    NB: if tuples list is empty, returns empty tuple, NOT tuple of empty lists.
    Use unzip2 for strict unzipping of list of pairs.
    """
    return tuple(map(list, zip(*tuples)))

def unzip2(pairs):
    """ :: list((a,b)) -> list(a), list(b) """
    if len(pairs) == 0: return [], []
    x, y = zip(*pairs)
    return list(x), list(y)

def for_each(f, it):
    for x in it: f(x)

# predicates

def not_none(x): return x is not None
def nonneg(x):   return x>=0
def positive(x): return x>0
def natural(x):  return x>=0 and int(x) == x
def not_in(y):   return lambda x: x not in y
def leq(y): return lambda x: x<=y
def geq(y): return lambda x: x>=y
def gt(y):  return lambda x: x>y
def lt(y):  return lambda x: x<y
def neq(y): return lambda x: x!=y
def eq(y):  return lambda x: x==y

def repeat_until_success_or_exhaustion(n_tries, criterion, description, action,
                                       initial_state=None,
                                       must_succeed=False):
    """ :: natural, (A->bool), string, (->A), A, bool -> A """
    ret = initial_state
    for _ in range(n_tries):
        ret = action()
        if criterion(ret):
            return ret
    if must_succeed:
        raise RuntimeError("Failed to %r after %r attempts" % (description, n_tries))
    else:
        return ret
