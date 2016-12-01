import numpy as np
import scipy.misc

def uniform_sample(x):
    """
    Samples from a list with uniform probability
    :param x: List of elements
    :return: A randomly sampled element from x under uniform distribution
    """
    return np.random.choice(x)

def binary_sample(x):
    """
    Takes an array of floats (probabilities) between zero and one and samples binary values for each entry
    :param x: np.array of floats between zero and one
    :return: np.array of same size filled with zeros and/or ones, which have been sampled from x
    """
    return np.random.binomial(1, p=x)


def logsumexp(p):
    """
    Slightly modified custom version of standard scipy logsumexp, which returns the logarithm of the sum of a list of exponentials.
    """
    return scipy.misc.logsumexp(p) if (len(p) > 0) else -1e20


def sample_uniformly(x):
    return x[int(len(x)*np.random.sample())]

def sample(x, p=None):
    """
    Randomly sample elements of a list, either from a uniform default distribution or from a user-defined probability list if this is defined.
    """
    s = np.random.random_sample()
    if p is None:
        return x[int(s*len(x))]
    else:
        p = np.cumsum(p)
        p = p / float(p[-1])
        return x[sum(s >= p)]


def sample_bool(p=.5):
    """
    Sample True or False with probability provided for True.
    """
    return bool(np.random.choice([True, False], p=[p, 1-p]))


def sample_dict(x):
    """
    Same as sample(), but using a dictionary of *option: probability* combinations.
    BAD FOR REPRODUCIBILITY!
    """
    return sample_unnormalised_pairs(x.items())

def sample_logdist(log_probs):
    return 0 if len(log_probs)==1 else sample_discrete1(logprobs_to_probs(log_probs))

def sample_normalised_pairs(pairs):
    return select_from_pairs_by_cumsum(pairs, np.random.random_sample())

def sample_unnormalised_pairs(pairs):
    return select_from_pairs_by_cumsum(pairs, sum(map(snd, pairs)) * np.random.random_sample())

def select_from_pairs_by_cumsum(pairs, u):
    tot=0
    for (v, p) in pairs:
        tot += p
        if u<tot:
            return v

def normalize_distribution(x):
    """
    Normalises a distribution.
    """
    x = np.array(x)
    s = np.sum(x)
    return x/s if (s > 0) else x

def maxnorm(x):
    return x - np.max(x)

def stoch(x):
    """ normalise probability distribution (for numpy arrays only) """
    return x/np.sum(x)

def entropy(x):
    """ entropy of normalised distribution in bits """
    nz = np.nonzero(x)[0]
    return -np.sum(x[nz]*np.log2(x[nz]))

def discrete_sampler(domain):
    sample = sample_discrete1 if len(domain)>50 else sample_discrete2
    return lambda probs: domain[sample(probs)]

# faster for large sample space
def sample_discrete1(probs):
    return np.searchsorted(np.cumsum(probs), np.random.random_sample())

# faster for small sample space
def sample_discrete2(probs):
    return select_from_pairs_by_cumsum(enumerate(probs), np.random.random_sample())

class RejectionSamplingFailure(Exception): pass

def repeat_until_not_none(num_tries, maybe_get_value, get_msg):
    """ :: nat, (void -> maybe(A)), (void -> str) -> A """
    for _ in range(num_tries):
        result = maybe_get_value()
        if result is not None: return result
    raise RejectionSamplingFailure("Rejection sampling failed after %d tries (%s)" % (num_tries, get_msg()))
