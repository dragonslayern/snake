import tensorflow as tf
import numpy as np

learning_rate = 0.0001

X = tf.placeholder(tf.float32, shape=[None,34,46,1], name="input")
Y = tf.placeholder(tf.float32, shape=[None, 4], name="loss")

conv1 = tf.layers.conv2d(X, filters=16, kernel_size=8, strides=[4, 4], padding="SAME", name="conv1")
conv1 = tf.layers.batch_normalization(conv1)
conv1 = tf.nn.relu(conv1)
conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=4, activation=tf.nn.relu, strides=[2, 2], padding="SAME", name="conv2")
conv2 = tf.layers.batch_normalization(conv2)
conv2 = tf.nn.relu(conv2)
conv2 = tf.reshape(conv2, [-1, 5*6*32], name="reshape")
# dense1 = tf.layers.dense(conv2, units=8, activation=tf.nn.relu, name="dense")
logits = tf.layers.dense(conv2, units=4, activation=tf.nn.relu, name="logits")

# dense1 = tf.layers.dense(X, units=32, activation=tf.nn.relu, name="dense1")
# dense2 = tf.layers.dense(dense1, units=128, activation=tf.nn.relu, name="dense2")
# logits = tf.layers.dense(dense1, units=4, activation=tf.nn.relu, name="logits")

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

prediction = tf.nn.softmax(logits, name="prediction")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

	# saver.restore(sess, 'model/snake_model.ckpt')
	sess.run(init)
	a = np.random.rand(34,46,1)
	a = a[np.newaxis,:]
	b = np.zeros((1,4))
	pred = sess.run(prediction, feed_dict={X: a})
	# b[0][np.argmax(pred)] = 1

	sess.run([train_op], feed_dict={X: a, Y: b})

	saver.save(sess, 'model/snake_model11.ckpt')