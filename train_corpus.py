#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script for training dutch corpora into word embeddings
#
# @author Baran Polat

import gensim
import logging
import os
import argparse
import multiprocessing as mp

# Parsers

parser = argparse.ArgumentParser(description='Script for training corpora for word vector models')
parser.add_argument('corpora', type=str, help='source folder with preprocessed corpora')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=300, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of sliding show')
parser.add_argument('-m', '--mincount', type=int, default=5, help='minimum number of of word occurrences for consideration')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train model')
parser.add_argument('-g', '--sg', type=int, default=0, help='training architecture: skip-gram=1, cbow=0')
parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('-n', '--negative', type=int, default=0, help='use negative sampling for training')
parser.add_argument('-o', '--cbowmean', type=int, default=0, help='for cbow training architecture: sum=0, mean=1 to merge context')
args = parser.parse_args()
logging.basicConfig(filename=args.target.strip() + '.result', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# get corpus sentences
class CorpusSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            with open(os.path.join(self.dirname, fname)) as fp:
                for line in fp:
                    yield line.split()

sentences = CorpusSentences(args.corpora)

# Train model
model=gensim.models.Word2Vec(
    sentences,
    size=args.size,
    window=args.window,
    min_count=args.mincount,
    workers=args.threads,
    sg=args.sg,
    hs=args.hs,
    negative=args.negative,
    cbow_mean=args.cbowmean
)
# store model
model.wv.save_word2vec_format(args.target, binary=True)
