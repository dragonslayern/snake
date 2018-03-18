import numpy as np 
import tensorflow as tf 
from collections import deque
from random import randint, uniform
from snake import snake

def update_nn(inputs, predictions, winner):

	print("Starting training")

	tf.reset_default_graph()

	learning_rate = 0.000001

	X = tf.placeholder(tf.float32, shape=[None, 34,46,1], name="input")
	Y = tf.placeholder(tf.float32, shape=[None, 4], name="loss")

	conv1 = tf.layers.conv2d(X, filters=16, kernel_size=8, strides=[4, 4], padding="SAME", name="conv1")
	conv1 = tf.layers.batch_normalization(conv1)
	conv1 = tf.nn.relu(conv1)
	conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=4, activation=tf.nn.relu, strides=[2, 2], padding="SAME", name="conv2")
	conv2 = tf.layers.batch_normalization(conv2)
	conv2 = tf.nn.relu(conv2)
	conv2 = tf.reshape(conv2, [-1, 5*6*32], name="reshape")
	logits = tf.layers.dense(conv2, units=4, activation=tf.nn.relu, name="logits")


	# dense1 = tf.layers.dense(X, units=32, activation=tf.nn.relu, name="dense1")
	# dense2 = tf.layers.dense(dense1, units=128, activation=tf.nn.relu, name="dense2")
	# logits = tf.layers.dense(dense1, units=4, activation=tf.nn.relu, name="logits")

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	prediction = tf.nn.softmax(logits, name="prediction")

	saver = tf.train.Saver()

	with tf.Session() as sess:

		saver.restore(sess, 'model/snake_model11.ckpt')

		for i in range(len(inputs)):

			x = np.array(inputs[i])
			# x = x[np.newaxis,:,]
			y = np.zeros((1,4))
			# y[0][3] = 1
			if i < len(inputs) - 2:
				y[0][predictions[i]] = 1
			else: 
				p = randint(0,3)
				while p == predictions[i]:
					p = randint(0,3)

				y[0][p] = 1

			sess.run([train_op], feed_dict={X: x, Y: y})

		saver.save(sess, 'model/snake_model11.ckpt')
		print("Training done. Model saved!")


def within_bounds(x, y, height, width):
	return x < height and x >= 0 and y < width and y >= 0

def tile_available(screen, x, y):
	return screen[x,y] == 0

width = 46
height = 34

arr = np.arange(1,6)
np.random.shuffle(arr)

s1 = snake("1")
s1.set_starting_pos(4,35)
snake1 = (4,35)

s2 = snake("2")
s2.set_starting_pos(21,3)
snake2 = (21,3)

s3 = snake("3")
s3.set_starting_pos(32,23)
snake3 = (32,23)

s4 = snake("4")
s4.set_starting_pos(4,10)
snake4 = (4,10)

s5 = snake("5")
s5.set_starting_pos(21,42)
snake5 = (21,42)

snakes = [s1, s2, s3, s4, s5]
obstacles = []

for i in range(50):
	obstacles.append((randint(0,height-1),randint(0,width-1)))

tick = 1
game_over = False

print("starting game!")

while not game_over:

	screen = np.zeros((height, width))
	
	for obs in obstacles:
		screen[obs] = 0.2
	
	snakes_alive = 0

	for snake in snakes:
		if snake.is_alive():
			snakes_alive += 1
			body = snake.get_body()
			for pos in body:
				screen[pos] = 0.5
			screen[snake.get_head()] = 0.6

	for snake in snakes:

		if snake.is_alive():

			direction = snake.get_direction(screen)

			if direction == "UP":

				new_pos_x = snake.get_head()[0]-1
				new_pos_y = snake.get_head()[1] 

				if not within_bounds(new_pos_x, new_pos_y, height, width) or not tile_available(screen, new_pos_x, new_pos_y):
					snake.on_snake_dead()
				else:
					snake.move_snake(new_pos_x,new_pos_y, tick)

			if direction == "DOWN":

				new_pos_x = snake.get_head()[0]+1
				new_pos_y = snake.get_head()[1] 

				if not within_bounds(new_pos_x, new_pos_y, height, width) or not tile_available(screen, new_pos_x, new_pos_y):
					snake.on_snake_dead()
				else:
					snake.move_snake(new_pos_x,new_pos_y, tick)

			if direction == "LEFT":

				new_pos_x = snake.get_head()[0] 
				new_pos_y = snake.get_head()[1]-1

				if not within_bounds(new_pos_x, new_pos_y, height, width) or not tile_available(screen, new_pos_x, new_pos_y):
					snake.on_snake_dead()
				else:
					snake.move_snake(new_pos_x,new_pos_y, tick)

			if direction == "RIGHT":

				new_pos_x = snake.get_head()[0]
				new_pos_y = snake.get_head()[1]+1 

				if not within_bounds(new_pos_x, new_pos_y, height, width) or not tile_available(screen, new_pos_x, new_pos_y):
					snake.on_snake_dead()
				else:
					snake.move_snake(new_pos_x,new_pos_y, tick)

	if snakes_alive <= 1:
		game_over = True

	tick += 1

print("Game over\n \n")
print("Ticks: " + str(tick) +  "\n \n")

for snake in snakes:
	if snake.is_alive():
		snake.won()
	snake.on_game_ended()

for snake in snakes:
	update_nn(snake.get_inputs(),snake.get_predictions(),snake.is_winner())
	






