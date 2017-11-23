from __future__ import print_function
from read_data import read_data
import tensorflow as tf



# Parameters
nb_classes = 2
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

def build(data, m, n, lambd):


    # Set model weights
    W = tf.Variable(tf.random_normal([n, nb_classes]), name='W')
    b = tf.Variable(tf.random_normal([nb_classes]), name='b')
    # regularization term
    regularizer = tf.nn.l2_loss(W) * lambd / batch_size
    # Construct model
    pred = tf.nn.softmax(tf.matmul(data, W) + b, name='pred') # Softmax


    return pred, regularizer

def train(X_train, X_test, y_train, y_test, lambd=1.0):

    m = X_train.shape[0]
    n = X_train.shape[1]

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n], name='x')
    y = tf.placeholder(tf.float32, [None, nb_classes], name='y')


    pred, regularizer = build(x, m, n, lambd)
    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + regularizer)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Test model
    predict_op = tf.argmax(pred, 1, name='predict_op')
    correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # initiate Saver object
    saver = tf.train.Saver()
    # Start training
    with tf.Session() as sess:

         # Run the initializer
        sess.run(init)

        # Training cycle
        step = 0
        for epoch in range(training_epochs):

            epoch_loss = 0
            i = 0


            while i < m:
                start = i
                end = i + batch_size
                batch_xs = X_train[start:end]
                batch_ys = y_train[start:end]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                epoch_loss += c
                i += batch_size
                step += 1

            print('Epoch', epoch, 'completed out of', training_epochs, 'loss:', epoch_loss)

        print("Optimization Finished!")

        print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
        # save trained network
        saver.save(sess, '../TENSORFLOW/my_model', global_step=step)

# Import data
X_train, X_test, y_train, y_test = read_data('train.csv')
train(X_train, X_test, y_train, y_test)