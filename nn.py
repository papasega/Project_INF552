import tensorflow as tf
from read_data import read_data

X_train, X_test, y_train, y_test = read_data('train.csv')
m = X_train.shape[0]
nb_features = X_train.shape[1]

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000
n_nodes_hl5 = 1000

n_classes = 2
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')


def neural_network_model(data, lambd):

    print('nb of features : {}'.format(nb_features))

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([nb_features, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    # regularization term
    regularizer = 0

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    regularizer += tf.nn.l2_loss(hidden_1_layer['weights'])

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    regularizer += tf.nn.l2_loss(hidden_2_layer['weights'])

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    regularizer += tf.nn.l2_loss(hidden_3_layer['weights'])

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    regularizer += tf.nn.l2_loss(hidden_4_layer['weights'])

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)
    regularizer += tf.nn.l2_loss(hidden_5_layer['weights'])

    output = tf.matmul(l5, output_layer['weights']) + output_layer['biases']
    regularizer += tf.nn.l2_loss(output_layer['weights'])
    regularizer *= lambd/m

    return output, regularizer


def train_neural_network(x, lambd=3.0):
    prediction, regularizer= neural_network_model(x, lambd)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(loss + regularizer)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_size
                batch_x = X_train[start:end]
                batch_y = y_train[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))

print('\nStart training neural network ... \n')
train_neural_network(x)