import tensorflow as tf
import numpy as np
from read_data import read_data

import pandas as pd
import math as mt
def predict_proba(X_):
    with tf.Session() as sess:
        # restore session variables and operators from file
        saver = tf.train.import_meta_graph('my_model-1428600.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()

        # retake place to feed new data
        x=graph.get_tensor_by_name('x:0')
        feed_dict={x:X_}

        # retrieve probability operator
        probs = graph.get_tensor_by_name('probs:0')
        pred = sess.run(probs,feed_dict)
        return pred

def error_rate(p, t):
    return np.mean(p != t)

def test_submission(filename):
    print('\nread and preprocess data ...\n')
    data = pd.read_csv(filename)
    ids = data.iloc[:,0]
    print('id shape : {}'.format(np.shape(ids)))
    X = data.iloc[:,1:]
    m = X.shape[0]
    n = X.shape[1]
    for i in range(n):
        indices = np.where(X.values[:,i] != -1)[0]
        mean = np.mean(X.values[indices, i])
        var = np.var(X.values[indices, i])
        X.values[np.where(X.values[:,i] == -1)[0], i] = np.random.normal(mean, mt.sqrt(var), m - indices.shape[0])
        X.values[:,i] = (X.values[:,i] - np.min(X.values[:,i])) / (np.max(X.values[:,i]) - np.min(X.values[:,i]))

    print('\nretrieve proba from regression nn...\n')
    pred = predict_proba(X.values)
    print('row 0 : {}'.format(pred[0]))
    print('row 1 : {}'.format(pred[1]))
    print('shape pred : {}'.format(np.shape(pred)))
    assert(pred.shape[0]==m)

    print('\noutput to .csv file ...\n')
    df = pd.DataFrame({"id": ids, "target": pred[:,1]})
    df.to_csv("../TENSORFLOW/submission.csv", index=False)
    print('\nDone!\n')

test_submission('test.csv')