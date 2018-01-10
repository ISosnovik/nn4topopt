import numpy as np


def uniform_sampler():
    return lambda: np.random.randint(1, 99)


def poisson_sampler(lam):
    def func():
        iter_ = max(np.random.poisson(lam), 1)
        iter_ = min(iter_, 99)
        return iter_
    return func


def parse_sampler(descriptor):
    descriptor = descriptor.lower()
    
    if descriptor == 'uniform':
        return uniform_sampler()
    
    try:
        name, lam = descriptor.split('_')
        lam = float(lam)
        if name == 'poisson':
            return poisson_sampler(lam)
    except:
        pass
    
    print('Argument `descriptor` should be either '
          '"uniform" or "poisson_[LAM]"\n'
          '--LAM: Lambda parameter in Poisson distribution')
