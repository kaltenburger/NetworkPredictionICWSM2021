# 07/28/2018
from __future__ import division
import argparse
from collections import Counter
import csv
#import jgraph
import itertools
import json
import matplotlib.pyplot as plt
import networkx as nx
from numpy import save as np_save
import numpy as np
import pandas as pd
import pickle
import os
from os import listdir
from os.path import join as path_join
import random
import re
from scipy.stats import norm
from scipy.stats import bernoulli
import scipy
import scipy.special
import seaborn
seaborn.set_style(style='white')
import sklearn
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
#import statsmodels.api as sm
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
#from sklearn import cross_validation, datasets, linear_model
from scipy.stats import chisquare
import sys

## function to create + save dictionary of features
def create_dict(key, obj):
    return(dict([(key[i], obj[i]) for i in range(len(key)) ]))

def save_dict(feature_name, obj):
    with open(feature_name, 'w') as outfile:
        json.dump(obj, outfile)
