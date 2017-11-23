import tensorflow as tf
import numpy as np
from read_data import read_data


def predict(X_, y_):
    with tf.Session() as sess:
        # restore session variables and operators from file
        saver = tf.train.import_meta_graph('my_model-47620.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()

        # retake place to feed new data
        x=graph.get_tensor_by_name('x:0')
        y=graph.get_tensor_by_name('y:0')
        feed_dict={x:X_,y:y_}

        # retrieve accuracy operator
        predict_op = graph.get_tensor_by_name('predict_op:0')

        accuracy = graph.get_tensor_by_name('accuracy:0')
        probs = graph.get_tensor_by_name('probs:0')
        p = sess.run(accuracy,feed_dict)
        pred = sess.run(probs,feed_dict={x:X_})
        res = sess.run(predict_op, feed_dict={x:X_})
        sum = np.sum(res == 0)
        m = np.shape(res)[0]
        print('result : {} - {}'.format(sum, m))
        print('\n1 row : {}\n'.format(pred[0]))
        print('\n2 row : {}\n'.format(pred[1]))
        print('accuracy : {}'.format(p))

def error_rate(p, t):
    return np.mean(p != t)

X_train, X_test, y_train, y_test = read_data('train.csv')
predict(X_test, y_test)