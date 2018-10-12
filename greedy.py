import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.externals.joblib.parallel import Parallel, delayed
import itertools
import math
import random
import scipy.sparse
import boto3, botocore
import multiprocessing
import logging

logging.basicConfig(
    format='%(asctime)s: %(message)s',
    level='INFO',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='greedy.log',
    filemode='w')

# logging.info num cores
num_cores = multiprocessing.cpu_count()
logging.info('num_cores:')
logging.info(num_cores)

logging.info('Fetching Files')
# bio regression experiment
foldername = 'FOR_YARON/'
filename = 'all_tissues_batch1_20180908_log_tpm.txt'
# read in feature matrix
nrow = 10633
ncol = 8979
parallel = True
# init matrix
X_train = np.zeros((nrow, ncol))

# fill feature space
file = open(foldername + filename, "r")

rows = []
cols = []
vals = []
for aline in file.readlines():
    values = aline.split()
    rows.append(int(values[0]) - 1)
    cols.append(int(values[1]) - 1)
    vals.append(float(values[2]))

file.close()

X = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(nrow, ncol))
X = X.T
X.shape

# import y
filename = 'all_tissues_batch1_20180908_meta.txt'

file = open(foldername + filename, "r")

y_cat = []
for aline in file.readlines():
    values = aline.split()
    y_cat.append(values[4])
file.close()

logging.info('Finished Parsing Files')


def get_class_rate(x_t, y_t):
    # Create logistic regression object
    logitm = LogisticRegression(C=100)
    logitm.fit(x_t, y_t)
    y_logit = logitm.predict(x_t)
    class_rate = np.sum(y_logit == y_t) / len(y_t)
    return class_rate


X1 = scipy.sparse.csr_matrix(X)

X1_cols = np.load('cols_to_use_2.npy')

all_predictors = X1_cols


def run_greedy(kToSelect, parallel):
    # get starting set of data
    predictors = [([], -1e10)]
    # loop through predictors and at each step,
    # add one predictor that increases R2 the most
    # and calculate R2
    for k in range(kToSelect):
        logging.info(k)
        best_k_predictors = predictors[-1][0]

        predictor_list = list(set(all_predictors) - set(best_k_predictors))

        def greedy_helper():
            k_plus_1 = list(best_k_predictors + [predictor])
            x_train = X1[:, k_plus_1]
            return get_class_rate(x_train, y_cat)

        if parallel:
            r2 = Parallel(
                n_jobs=-1, verbose=50)(
                    delayed(greedy_helper)() for predictor in predictor_list)
        else:
            r2 = []
            for predictor in predictor_list:
                r2.append(greedy_helper())

        best_k_plus_1 = best_k_predictors + [predictor_list[np.argmax(r2)]]
        predictors.append((best_k_plus_1, np.max(r2)))
        logging.info(k, best_k_plus_1, np.max(r2))
    return predictors


logging.info('START')
predictors = run_greedy(10, parallel)
logging.info('END')
import pickle as pkl
pickle.dump(predictors, 'greedy.p')