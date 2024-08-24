import numpy as np
import pandas as pd


# %% Snippet 20.5
def linParts(numAtoms, numThreads):
    """Partition of atoms with a single loop"""
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# %% Snippet 20.6 The nestedParts Function
def nestedParts(numAtoms, numThreads, upperTriang=False):
    """Partition of atoms with an inner loop"""
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are the heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


# %% Snippet 20.7
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    """
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kargs: any other argument needed by func

    Example: df1 = mpPandasObj(func, ('molecule', df0.index), 24, **kargs)

    """
    import pandas as pd
    argList = pdObj  # winsm
    if linMols:
        parts = linParts(len(argList[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(argList[1]), numThreads * mpBatches)
    #
    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]: parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    #
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    #
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    #
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0


# %% Snippet 20.8 Single-Thread Execution, for Debugging
def processJobs_(jobs):
    """Run jobs sequentially, for debugging"""
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


# %% Snippet 20.9 Example of Asynchronous Call to Python's Multiprocessign Library
import multiprocessing as mp
import datetime as dt
import time
import sys


def reportProgress(jobNum, numJobs, time0, task):
    """Report progress as asynch jobs are completed"""
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    if jobNum < numJobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')
    return


def processJobs(jobs, task=None, numThreads=24):
    """Run in parallel."""
    # Jobs must contain a 'func' callback, for expandCall
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expandCall, jobs), [], time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close();
    pool.join()  # this is needed to prevent memory leaks
    return out


# %% Snippet 20.10 Passing the Job (Molecule) to the Callback Function
def expandCall(kargs):
    """Expand the arguments of a callback function, kargs['func']"""
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


# %% Snippet 20.11 Place this Code at the Beginning of Your Engine
# The instructions that should be listed at the top of your multiprocessing
# engine library. If you are curious about the precise reason this piece of
# code is needed, you may want to read Ascher et al. [2005], Section 7.5.
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


import copy_reg, types, multiprocessing as mp

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


# %% Snippet 20.12 Enhancing `processJobs` to Perform On-the-Fly Output Reduction
import copy


def processJobsRedux(jobs, task=None, numThreads=24, redux=None, reduxArgs={}, reduxInPlace=False):
    """
    Run in parallel
    jobs must contain a 'func' callback, for expandCall
    redux prevents wasting memory by reducing output on the fly
    """
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    imap, out, time0 = pool.imap_unordered(expandCall, jobs), None, time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(imap, 1):
        if out is None:
            if redux is None:
                out, redux, reduxInPlace = [out_], list.append, True
            else:
                out = copy.deepcopy(out_)
        else:
            if reduxInPlace:
                redux(out, out_, **reduxArgs)
            else:
                out = redux(out, out_, **reduxArgs)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out


# %% Snippet 20.13 Enhancing `mpPandasObj` to Perform On-the-Fly Output Reduction
def mpJobList(func, argList, numThreads=24, mpBatches=1, linMols=True, redux=None,
              reduxArgs={}, reduxInPlace=False, **kargs):
    if linMols:
        parts = linParts(len(argList[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(argList[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {argList[0]: argList[1][parts[i-1]: parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    out = processJobsRedux(jobs, redux=redux, reduxArgs=reduxArgs,
                           reduxInPlace=reduxInPlace, numThreads=numThreads)
    return out


# %% Snippet 20.14 Principal Compnents for a Subset of the columns
def getPCs(path, molecules, eVec):
    """get principal components by loading one file at a time"""
    pcs = None
    for i in molecules:
        df0 = pd.read_csv(path + i, index_col=0, parse_dates=True)
        if pcs is None:
            pcs = np.dot(df0.values, eVec.loc[df0.columns].values)
        else:
            pcs += np.dot(df0.values, eVec.loc[df0.columns].values)
    pcs = pd.DataFrame(pcs, index=df0.index, columns=eVec.columns)
    return pcs


def main():
    pcs = mpJobList(getPCs)


if __name__ == '__main__':
    main()
