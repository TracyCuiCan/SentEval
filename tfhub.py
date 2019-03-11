from __future__ import absolute_import, division, unicode_literals

import sys
import os
import logging
import tensorflow as tf
import tensorflow_hub as hub
tf.logging.set_verbosity(tf.logging.INFO)


# Set PATHs
PATH_SENTEVAL = ''
PATH_TO_DATA = './data'

# TF-Hub modules
MODULES = [
    'https://tfhub.dev/google/universal-sentence-encoder-large/3',
    'https://tfhub.dev/google/Wiki-words-500/1',
    #'https://tfhub.dev/google/nnlm-en-dim128/1',
    #'https://tfhub.dev/google/elmo/2',
]


# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params['module'](batch)
    return embeddings

def make_embed_fn(module):
  with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    embed = hub.Module(module)
    embeddings = embed(sentences)
    session = tf.train.MonitoredSession()
  return lambda x: session.run(embeddings, {sentences: x})

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
  # Set up logger
  for mdl in MODULES:
    print("*-----------------------------------------------------------------*")
    print("Evaluating module: " + mdl)
    print("*-----------------------------------------------------------------*")

    module = make_embed_fn(mdl)
    params_senteval['module'] = module

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ['STS15', 'STS16', 'MR', 'CR', 'SUBJ']
    results = se.eval(transfer_tasks)
    print(results)
