# -*- coding: utf-8 -*-

# Python file to produce gender biases
#
# @author Baran Polat

from __future__ import print_function, division
import json
import gensim
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot  as plt



# from Bolukbasi et al.
# Used to create a gender direction
def doPCA(pairs, embedding, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (embedding[a] + embedding[b]) / 2
        matrix.append(embedding[a] - center)
        matrix.append(embedding[b] - center)

    matrix = np.array(matrix)
    pca = PCA(n_components=num_components)
    pca.fit(matrix)
    return pca

# Calculate bias per word
def genderBias(genderNeutralWords, genderdirection, embedding):
    sp = sorted([(embedding[w].dot(genderdirection), w) for w in genderNeutralWords])
    return sp


