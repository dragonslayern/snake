import logging
import util
import numpy as np
from random import randint
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import deepcopy

log = logging.getLogger("client.snake")


class Snake(object):


	def __init__(self):
		self.name = "Love"
		self.snake_id = None
		self.last_direction = None
		self.last_prediction = None

		self.checkpoint_path = "model/dqg.ckpt"
		self.X_state = tf.placeholder(tf.float32, shape=[None, 34, 46, 1])
		self.actor_q_values, actor_vars = self.q_network(self.X_state, name="q_networks/actor")
		self.saver = tf.train.Saver()
		self.sess = tf.Session()    
		self.saver.restore(self.sess, self.checkpoint_path)

	def q_network(self, X_state, name):

		learning_rate = 0.0001
		input_height = 34
		input_width = 46
		input_channels = 1
		conv_n_maps = [32, 32, 32]
		conv_kernel_strides = [(3,3), (3,3), (3,3)]
		conv_strides = [1,1,1]
		conv_padding = ["Same"] * 3
		conv_activation = [tf.nn.relu] * 3
		n_hidden_in = 34*46*32
		n_hidden = 64
		hidden_activation = tf.nn.relu
		n_ouputs = 4
		initializer = tf.contrib.layers.variance_scaling_initializer()

		prev_layer = X_state
		conv_layers = []
		with tf.variable_scope(name) as scope:
			for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_strides, conv_strides, conv_padding, conv_activation):
				prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size, 
											  strides=stride, padding=padding, activation=activation, kernel_initializer=initializer)
			last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
			hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation, kernel_initializer=initializer)
			outputs = tf.layers.dense(hidden, n_ouputs, kernel_initializer=initializer)
		trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
		trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
		return outputs, trainable_vars_by_name

	def get_next_move(self, game_map):

		# print(game_map.game_map)

		value_other_snakes = 0.4
		value_other_snakes_head = 0.5
		value_my_snake = 0.9
		value_my_snake_head = 1.0
		value_obstacles = 0.4

		screenSize = ([game_map.height, game_map.width])
		screen = np.zeros(screenSize)

		for snake in game_map.game_map['snakeInfos']:
			first = True
			for positions in snake['positions']:
				value = value_other_snakes
				if snake['id'] == self.snake_id:
					value = value_my_snake
				y,x = util.translate_position(positions, game_map.width)
				if first:
					screen[x,y] = value_other_snakes_head
					first = False
				else:
					screen[x,y] = value

		if not type(game_map.game_map['obstaclePositions']) == int:
			for position in game_map.game_map['obstaclePositions']:
				y,x = util.translate_position(position, game_map.width)
				screen[x,y] = value_obstacles
		else: 
			y,x = util.translate_position(game_map.game_map['obstaclePositions'], game_map.width)
			screen[x,y] = value_obstacles

		snake = game_map.get_snake_by_id(self.snake_id)
		y,x = util.translate_position(snake['positions'][0], game_map.width)
		screen[x,y] = value_my_snake_head
		
		self.last_screen = screen

		direction, longest = self.longest_path(x,y, game_map)

		if longest < 15:

			prediction = self.get_prediction(screen)

			if prediction == 0:
				if game_map.can_snake_move_in_direction(self.snake_id, util.Direction.UP):
					direction = util.Direction.UP
				else:
					direction, _ = self.longest_path(x, y, game_map)
			elif prediction == 1:
				if game_map.can_snake_move_in_direction(self.snake_id, util.Direction.DOWN):
					direction = util.Direction.DOWN
				else:
					direction, _ = self.longest_path(x, y, game_map)
			elif prediction == 2:
				if game_map.can_snake_move_in_direction(self.snake_id, util.Direction.LEFT):
					direction = util.Direction.LEFT
				else:	
					direction, _ = self.longest_path(x, y, game_map)
			elif prediction == 3:
				if game_map.can_snake_move_in_direction(self.snake_id, util.Direction.RIGHT):
					direction = util.Direction.RIGHT
				else:
					direction, _ = self.longest_path(x, y, game_map)

		return direction
		

	def longest_path(self, x, y, game_map):

		direction = None
		longest = -1

		longest_left = self.check_path(x, y-1, deepcopy(self.last_screen), 0, game_map)

		if longest_left > longest:
			longest = longest_left
			direction = util.Direction.LEFT

		longest_right = self.check_path(x, y+1, deepcopy(self.last_screen), 0, game_map)

		if longest_right > longest:
			longest = longest_right
			direction = util.Direction.RIGHT

		longest_up = self.check_path(x-1, y, deepcopy(self.last_screen), 0, game_map)

		if longest_up > longest:
			longest = longest_up
			direction = util.Direction.UP

		longest_down = self.check_path(x+1, y, deepcopy(self.last_screen), 0, game_map)

		if longest_down > longest:
			longest = longest_down
			direction = util.Direction.DOWN

		return direction, longest

	def check_path(self, x, y, visited, depth, game_map):

		if depth == 30:
			return depth

		if game_map.is_coordinate_out_of_bounds((y,x)) or visited[x,y] != 0:
			return depth

		visited[x, y] = 1

		if not game_map.is_coordinate_out_of_bounds((y,x-1)) and visited[x-1, y] == 0:
		 	return self.check_path(x-1, y, visited, depth+1, game_map)
		if not game_map.is_coordinate_out_of_bounds((y,x+1)) and visited[x+1, y] == 0:
			return self.check_path(x+1, y, visited, depth+1, game_map)
		if not game_map.is_coordinate_out_of_bounds((y-1,x)) and visited[x, y-1] == 0:
			return self.check_path(x, y-1, visited, depth+1, game_map)
		if not game_map.is_coordinate_out_of_bounds((y+1,x)) and visited[x, y+1] == 0:
	 		return self.check_path(x, y+1, visited, depth+1, game_map) 

		return depth

	def get_prediction(self, screen):

		q_values = self.sess.run([self.actor_q_values], feed_dict={self.X_state: screen[np.newaxis,:,:, np.newaxis]})
		action = np.argmax(q_values)

		return action

	def on_game_ended(self):
		strFile = "last_game/test.png"
		plt.imshow(self.last_screen)
		plt.savefig(strFile)
		log.debug('The game has ended!')

	def on_snake_dead(self, reason):
		log.debug('Our snake died because %s', reason)

	def on_game_starting(self):
		log.debug('Game is starting!')		

	def on_player_registered(self, snake_id):
		log.debug('Player registered successfully')
		self.snake_id = snake_id

	def on_invalid_player_name(self):
		log.fatal('Player name is invalid, try another!')

	def on_game_result(self, player_ranks):
		log.info('Game result:')
		for player in player_ranks:
			is_alive = 'alive' if player['alive'] else 'dead'
			log.info('%d. %d pts\t%s\t(%s)' %
					(player['rank'], player['points'], player['playerName'],
					is_alive))


def get_snake():
	return Snake()