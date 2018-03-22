import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

class snake(object):

	def __del__(self):
		print("Snake Removed")

	def __init__(self, name):
		self.name = name
		self.alive = True
		self.body = deque()
		self.screenshots = []
		self.bot = False

	def reset(self):
		self.alive = True
		self.body = deque()
		self.screenshots = []
	
	def set_starting_pos(self,x,y):
		self.body.append((x,y))

	def get_state(self, screen):

		screen = deepcopy(screen)
		for pos in self.body:
			screen[(pos)] = 0.8
		screen[self.get_head()] = 1.0
		self.screenshots.append(screen)
		return screen

	def kill(self):
		self.alive = False
		# print("Snake dead: " + self.name)

	def is_bot(self):
		return self.bot

	def set_bot(self):
		self.bot = True

	def print_game(self):

		if self.winner:
			folder = 'last_game'
			for the_file in os.listdir(folder):
			    file_path = os.path.join(folder, the_file)
			    try:
			        if os.path.isfile(file_path):
			            os.unlink(file_path)
			    except Exception as e:
			        print(e)

			print("Snake " + self.name + " won! FUCK YEAH")
			print("Saving last game")
			for i, s in enumerate(self.screenshots):
				strFile = "last_game/" + str(i) + '.png'
				if os.path.isfile(strFile):
				   os.remove(strFile)
				plt.imshow(s)
				plt.savefig(strFile)

	def get_body(self):
		return self.body

	def get_head(self):
		return self.body[0]

	def is_alive(self):
		return self.alive

	# def won(self):
		# print("Snake " + self.name + " WON!!!!!")
		# self.print_game()

	def move_snake(self, x, y, tick):
		if not (tick % 3) == 0:
			if tick > 3:
				self.body.pop()
		
		self.body.appendleft((x,y))

