import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from collections import deque
from random import randint
from copy import deepcopy

class snake(object):

	def __init__(self, name):
		self.name = name
		self.alive = True
		self.winner = False
		self.body = deque()
		self.screenshots = []
		self.predictions = []
		self.inputs = []

		self.sess = tf.Session()    
		self.saver = tf.train.import_meta_graph('model/snake_model11.ckpt.meta')
		self.saver.restore(self.sess, tf.train.latest_checkpoint('model/'))

		self.graph = tf.get_default_graph()
		self.input_screen = self.graph.get_tensor_by_name("input:0")
		self.prediction_op = self.graph.get_tensor_by_name("prediction:0")
	
	def set_starting_pos(self,x,y):
		self.body.append((x,y))

	def get_direction(self, screen):

		screen = deepcopy(screen)
		for pos in self.body:
			screen[(pos)] = 0.8

		screen[self.get_head()] = 1.0

		# grid = np.zeros((11,11))

		# for i in range(-5,6):
		# 	for j in range(-5,6):
		# 		x,y = self.get_head()
		# 		curr_x = x + i
		# 		curr_y = y + j
		# 		if curr_x > 33 or curr_x < 0 or curr_y < 0 or curr_y > 45:
		# 			grid[i+5, j+5] = 1
		# 		elif not screen[curr_x, curr_y] == 0:
		# 			grid[i+5, j+5] = 1

		direction = "UP"

		self.screenshots.append(screen)
		prediction = self.get_prediction(screen)
		
		if prediction == 0:
			direction = "LEFT"
		if prediction == 1:
			direction = "UP"
		if prediction == 2:
			direction = "RIGHT"
		if prediction == 3:
			direction = "DOWN"

		return direction

	def get_prediction(self, screen):

		# x = np.zeros((34,46,4))
		# if len(self.screenshots) < 4:
		# 	for i in range(1,len(self.screenshots)+1):
		# 		x[:,:,i-1] = np.array(self.screenshots[-i])
		# else:
		# 	x[:,:,0] = np.array(self.screenshots[-1])
		# 	x[:,:,1] = np.array(self.screenshots[-2])
		# 	x[:,:,2] = np.array(self.screenshots[-3])
		# 	x[:,:,3] = np.array(self.screenshots[-4])

		# x = x[np.newaxis,:,:,:]

		# x = np.reshape(screen, 121)
		x = screen[np.newaxis,:,:,np.newaxis]
		feed_dict ={self.input_screen: x}
		prediction = self.sess.run([self.prediction_op], feed_dict)
		prediction = np.array(prediction)
		self.predictions.append(np.argmax(prediction))
		self.inputs.append(x)
		print(prediction)
		return np.argmax(prediction)

	def on_snake_dead(self):
		self.alive = False
		print("Snake dead: " + self.name)

	def on_game_ended(self):
		
		self.sess.close()

		# if self.winner:
		# 	folder = 'last_game'
		# 	for the_file in os.listdir(folder):
		# 	    file_path = os.path.join(folder, the_file)
		# 	    try:
		# 	        if os.path.isfile(file_path):
		# 	            os.unlink(file_path)
		# 	    except Exception as e:
		# 	        print(e)

		# 	print("Snake " + self.name + " won! FUCK YEAH")
		# 	print("Saving last game")
		# 	for i, s in enumerate(self.screenshots):
		# 		strFile = "last_game/" + str(i) + '.png'
		# 		if os.path.isfile(strFile):
		# 		   os.remove(strFile)
		# 		plt.imshow(s)
		# 		plt.savefig(strFile)

	def get_body(self):
		return self.body

	def get_reward(self):
		return self.reward

	def get_head(self):
		return self.body[0]

	def is_alive(self):
		return self.alive

	def won(self):
		self.winner = True

	def is_winner(self):
		return self.winner

	def get_predictions(self):
		return self.predictions

	def get_inputs(self):
		return self.inputs

	def get_screenshots(self):
		return self.screenshots

	def move_snake(self, x, y, tick):
		if not (tick % 3) == 0:
			if tick > 3:
				self.body.pop()
		
		self.body.appendleft((x,y))

