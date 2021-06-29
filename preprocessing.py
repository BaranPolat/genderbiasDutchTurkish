# -*- coding: utf-8 -*-

# script for preprocessing dutch corpora
#
# @author Baran Polat


# Libraries
import gensim
import nltk.data
from nltk.corpus import stopwords
import argparse
import os
import re
import logging
import sys
import multiprocessing as mp

# Parsers

parser = argparse.ArgumentParser(description='script for preprocessing dutch wikipedia corpus')
parser.add_argument('raw', type=str, help='source file with raw data for corpus creation')
parser.add_argument('target', type=str, help='target file name to store corpus in')
parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation tokens')
parser.add_argument('-s', '--stopwords', action='store_true', help='remove stopword tokens')
parser.add_argument('-b', '--bigram', action='store_true', help='detect and process common bigram phrases')
parser.add_argument('-t', '--threads', type=int, default=32, help='batch size for multiprocessing')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for multiprocessing')
parser.add_argument('-l', '--lowercase', action='store_true', help='turn tokens into lowercase')

args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#sentenceDetector = nltk.data.load('tokenizers/punkt/turkish.pickle')
#sentenceDetector = nltk.data.load('tokenizers/punkt/dutch.pickle')
punctuationTokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']',
                     '{', '}', '?', '!', '-', '–', '+', '*', '--', '\'\'', '``']

punctuation = '?.!/;:()&+'

#stop_words = stopwords.words('turkish')
#stop_words = stopwords.words('dutch')

def process_line(line):
    """
       Pre processes the given line.
       :param line: line as str
       :return: preprocessed sentence
    """
    # detect sentences
    sentences = sentenceDetector.tokenize(line)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if args.punctuation:
            words = [x for x in words if x not in punctuationTokens]
            words = [re.sub('[{}]'.format(punctuation), '', x) for x in words]
        if args.stopwords:
            words = [x for x in words if x not in stop_words]
        if args.lowercase:
            words = [x.lower() for x in words]
        if len(words) > 1:
            return '{}\n'.format(' '.join(words))





if not os.path.exists(os.path.dirname(args.target)):
    os.makedirs(os.path.dirname(args.target))
with open(args.raw, 'r') as infile:
    # preprocess with multiple threads
    pool = mp.Pool(args.threads)
    values = pool.imap(process_line, infile, chunksize=args.batch_size)
    with open(args.target, 'w') as outfile:
        for i, s in enumerate(values):
            if i and i % 25000 == 0:
                logging.info('processed {} sentences'.format(i))
                outfile.flush()
            if s:
                outfile.write(s)
        logging.info('preprocessing of {} sentences finished'.format(i))


class CorpusSentences:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


if args.bigram:
    logging.info('train bigram phrase detector')
    bigram = gensim.models.Phrases(CorpusSentences(args.target))
    logging.info('transform corpus to bigram phrases')
    with open('{}.bigram'.format(args.target), 'w') as outfile:
        for tokens in bigram[CorpusSentences(args.target)]:
            outfile.write('{}\n'.format(' '.join(tokens)))
    logging.info('bigrams finished')
