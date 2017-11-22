import tensorflow as tf
import numpy as np
from read_data import read_data

x = tf.placeholder('float')
y = tf.placeholder('float')
tf.reset_default_graph()
saver = tf.train.import_meta_graph('my_model.meta')
def predict(X_, y_):
    with tf.Session() as sess:

        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        print('accuracy : {}'.format(accuracy.eval({x:X_, y:y_})))

def error_rate(p, t):
    return np.mean(p != t)
X = 0
y = 0
predict(X, y)
X_train, X_test, y_train, y_test = read_data('train.csv')
predict(X_test, y_test)