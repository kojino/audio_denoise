import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import itertools
import math
import random
import scipy.sparse
from sklearn.externals.joblib.parallel import Parallel, delayed
import multiprocessing
import logging
#import boto3, botocore

logging.basicConfig(
    format='%(asctime)s: %(message)s',
    level='INFO',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='adaptive.log',
    filemode='w')

# logging.info num cores
num_cores = multiprocessing.cpu_count()
logging.info('num_cores:')
logging.info(num_cores)


def randomSample(X, num):
    # if not enough elements, just return X
    if len(X) < int(num):
        R = X
    else:
        R = [X[i] for i in sorted(random.sample(range(len(X)), int(num)))]
    return R


# alg 4
# estimateSet(all_predictors, ['lower'])
def estimateSet(X, S, m=5):
    est = 0
    fS = oracle(S)
    # repeat m times
    for it in range(m):
        # sample size k/r m times
        R = randomSample(X, k / r)
        est += oracle(R + S)
    return (est - m * fS) / m


# alg 5
def estimateMarginal(X, S, a, m=5):
    est = 0
    # repeat m times
    for it in range(m):
        # if there are not enough elements
        R = randomSample(X, k / r)
        marg1 = oracle(R + S + [a])
        if a in R:
            R.remove(a)
            marg2 = oracle(S + R)
        else:
            marg2 = oracle(S + R)
        est += marg1 - marg2
    return est / m


def get_class_rate(x_t, y_t):
    # Create logistic regression object
    logitm = LogisticRegression()
    logitm.fit(x_t, y_t)
    y_logit = logitm.predict(x_t)
    class_rate = np.sum(y_logit == y_t) / len(y_t)
    return class_rate


# given set of features, return r2
def oracle(cols):
    if cols == []:
        return 0.0
    else:
        r2 = get_class_rate(X1[:, cols], y_cat)
        return r2


def union(A, B):
    return list(set(A + B))


# alg 3/6
def amortizedFilter(k, r, ep, OPT, X, debug=True, parallel=False):

    m = 10
    S = []
    y_adap = []
    for i in range(r):
        T = []
        logging.info('r=' + str(i))

        fS = oracle(S)
        fST = oracle(union(S, T))

        while ((fST - fS) < (ep / 20) * (OPT - fS)) and (len(union(S, T)) < k):

            # FILTER Step
            # this only changes X
            vs = estimateSet(X, union(S, T), m)
            while (vs < (1 - ep) * (OPT - fST) / r):
                if debug:
                    logging.info('inner while loop')

                # get marginal contribution
                if parallel:
                    marg_a = Parallel(
                        n_jobs=-1, verbose=50)(
                            delayed(estimateMarginal)(X, union(S, T), a, m)
                            for a in X)
                else:
                    marg_a = [
                        estimateMarginal(X, union(S, T), a, m) for a in X
                    ]

                # Filter!
                Xnew = [
                    X[idx] for idx, el in enumerate(marg_a)
                    if el >= (1 + ep / 2) * (1 - ep) * (OPT - fST) / k
                ]
                X = Xnew

                # estimate if filtered set is good enough
                vs = estimateSet(X, union(S, T), m)
                if debug:
                    logging.info('Elements remaining: ' + str(len(X)))
                    logging.info('Check')
                    logging.info(vs < (1 - ep) * (OPT - fST) / r)

            R = randomSample(X, k / r)
            T = union(T, R)

            # T changes but S doesn't
            fST = oracle(union(S, T))

            if debug:
                logging.info('Outer Loop')
                logging.info(fST)

        S = union(S, T)
        fS = oracle(S)
        y_adap.append((len(S), fS))
    return y_adap


# AN EXAMPLE
logging.info('Fetching Files')

X1 = scipy.sparse.load_npz('x_data.npz')
y_cat = np.load('y_data.npy')

all_predictors = list(range(X1.shape[1]))
logging.info('Num Features: ' + str(len(all_predictors)))

logging.info('Starting Adaptive')

k = 50
r = 2
ep = 0.01
OPT = 0.5
y_adap = amortizedFilter(k, r, ep, OPT, all_predictors, parallel=True)
logging.info(y_adap)
