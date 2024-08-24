# %% Snippet 20.3
import numpy as np


def barrierTouch(r, width=.5):
    """Find the index of the earliest barrier touch"""
    t, p = {}, np.log((1+r).cumprod(axis=0))
    for j in range(r.shape[1]):  # go thorough columns
        for i in range(r.shape[0]): # go through rows
            if p[i, j] >= width or p[i, j] <= -width:
                t[j] = i
                continue
    return t


def main0():
    """Path dependency: Sequential implementation"""
    r = np.random.normal(0, .01, size=(1000, 10000))
    t = barrierTouch(r)
    return


if __name__ == '__main__':
    import timeit
    print(min(timeit.Timer('main0()', setup='from __main__ import main0').repeat(5, 10)))
