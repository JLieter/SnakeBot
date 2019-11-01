import pygame
import sys
from random import randint, randrange
import time
import math
import numpy as np
import random
import tensorflow as tf
import math

if len(sys.argv) == 4:
	DISPLAY = int(sys.argv[1])
	POPULATION = int(sys.argv[2])
	LOG = int(sys.argv[3])
else:
	LOG = 1
	DISPLAY = 1
	POPULATION = 25

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (100,100,100)
GREEN = (0,255,0)
WHITE = (255,255,255)

SPEED = 1
SCORE = 0
STARVATION_RATE = 200
if SCORE == 20:
	STARVATION_RATE = 500
MUTATION_RATE = 0.1
BLOCK_SIZE = 20
SCREEN_SIZE = 800
GEN = 0
SNAKE_COLOR = GREEN
TAIL_COLOR = GREY
FOOD_COLOR = "Random"
# FOOD_COLOR = RED
SCREEN_COLOR = BLACK
FOOD = False

pygame.init()
if DISPLAY:
	gameDisplay = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))


#######################MAKE SEPERATE FILE############################

class Snake:
	def __init__(self, Brain=None):
		self.DISPLAY = True
		self.x = SCREEN_SIZE/4
		self.y = SCREEN_SIZE/8
		self.x_speed = 1
		self.y_speed = 0
		self.tail_size = 0
		self.tail = []
		self.Direction = "RIGHT"
		self.SCORE = 0
		self.fitness = 0
		self.hunger = 0
		if Brain:
			self.Brain = Brain
		else:
			self.Brain = Neural_Network(5, 8, 3)
			
	def think(self, food):
		inputs = [(self.check_obstacle_front()),
				  (self.check_obstacle_left()),
				  (self.check_obstacle_right()),
				   food.x, 
				   food.y]
		pred = self.Brain.predict(inputs)
		return pred - 1

	def update(self):
		self.hunger += 1
		self.tail.insert(0, (self.x, self.y))
		if self.tail_size != len(self.tail):
			self.tail.pop()

		self.x += (self.x_speed*BLOCK_SIZE)
		self.y += (self.y_speed*BLOCK_SIZE)

	def show(self):
		for i in range(self.tail_size):
			pygame.draw.rect(gameDisplay, TAIL_COLOR, [self.tail[i][0], self.tail[i][1], BLOCK_SIZE, BLOCK_SIZE])

		pygame.draw.rect(gameDisplay, SNAKE_COLOR, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])

	def direction(self, x, y):
		if self.x_speed != -x:
			self.x_speed = x
		if self.y_speed != -y:
			self.y_speed = y

		if x == -1:
			self.Direction = "LEFT"
		if x == 1:
			self.Direction =  "RIGHT"
		if y == -1:
			self.Direction =  "UP"
		if y == 1:
			self.Direction =  "DOWN"

	def eats(self, pos_x, pos_y):
		if self.x == pos_x and self.y == pos_y:
			self.tail_size += 1
			return True

	def move_decision(self, keystroke):
		if self.Direction == "LEFT":
			if keystroke == -1:
				self.direction(0, -1)
			if keystroke == 1:
				self.direction(0, 1)
		elif self.Direction == "RIGHT":
			if keystroke == -1:
				self.direction(0, 1)
			if keystroke == 1:
				self.direction(0, -1)
		elif self.Direction == "UP":
			if keystroke == -1:
				self.direction(-1, 0)
			if keystroke == 1:
				self.direction(1, 0)
		elif self.Direction == "DOWN":
			if keystroke == -1:
				self.direction(1, 0)
			if keystroke == 1:
				self.direction(-1, 0)

	def check_collision(self):
        # Check if the head collides with the edges of the board
		if self.x >= (SCREEN_SIZE) or self.x < 0:
			return True
		elif self.y > (SCREEN_SIZE - BLOCK_SIZE) or self.y < 0:
			return True
        # Check if the head collides with the body
		for body in self.tail[1:]:
			if self.x == body[0] and self.y == body[1]:
				return True
		return False

	def starved(self):
		if self.hunger > STARVATION_RATE:
			return True
		return False		

	def check_obstacle(self):
		if self.x_speed == 0:
			one_step = self.y + (self.y_speed * BLOCK_SIZE)
			if one_step == 0 or one_step == SCREEN_SIZE-BLOCK_SIZE or (self.x, one_step) in self.tail:
				return 1
		elif self.y_speed == 0:
			one_step = self.x + (self.x_speed * BLOCK_SIZE)
			if one_step == 0 or one_step == SCREEN_SIZE-BLOCK_SIZE or (one_step, self.y) in self.tail:
				return 1
		return 0

	def terminate(self, LOG):
		if (self.starved()):
			if LOG == 2:
				print("SNAKEBOT  |  SNAKE DIED: STARVATION. SCORE: " + str(self.SCORE))
			return True
		elif (self.check_collision()):
			if LOG == 2:
				print("SNAKEBOT  |  SNAKE DIED: COLLISION. SCORE: " + str(self.SCORE))
			return True
		return False

	def check_obstacle_left(self):
		if self.Direction == "RIGHT":
			one_step = self.y - BLOCK_SIZE
			return one_step
		elif self.Direction == "LEFT":
			one_step = self.y + BLOCK_SIZE
			return one_step
		elif self.Direction == "UP":
			one_step = self.y - BLOCK_SIZE
			return one_step
		elif self.Direction == "DOWN":
			one_step = self.y + BLOCK_SIZE
			return one_step/SCREEN_SIZE

	def check_obstacle_right(self):
		if self.Direction == "RIGHT":
			one_step = self.y + BLOCK_SIZE
			return one_step
		elif self.Direction == "LEFT":
			one_step = self.y - BLOCK_SIZE
			return one_step
		elif self.Direction == "UP":
			one_step = self.y + BLOCK_SIZE
			return one_step
		elif self.Direction == "DOWN":
			one_step = self.y - BLOCK_SIZE
			return one_step/SCREEN_SIZE

	def check_obstacle_front(self):
		if self.x_speed == 0:
			one_step = self.y + (self.y_speed * BLOCK_SIZE)
			return one_step
		elif self.y_speed == 0:
			one_step = self.x + (self.x_speed * BLOCK_SIZE)
			return one_step/SCREEN_SIZE

	def mutate(self, x):
		if (np.random.random() < MUTATION_RATE):
			offset = np.random.uniform(-1,1)
			newx = x + offset
			return newx
		else:
			return x


#####################################################################

class Food:
	def __init__(self):
		self.x = (randint(0,((SCREEN_SIZE-BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE)
		self.y = (randint(0,((SCREEN_SIZE-BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE)

	def update(self):
		if FOOD_COLOR == "Random":
			self.COLOR = (randrange(254)|64, randrange(254)|64, randrange(254)|64)
		else:
			self.COLOR = FOOD_COLOR

	def show(self):
		pygame.draw.rect(gameDisplay, self.COLOR, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])

#####################################################################

class ActivationFunction:
	def __init__(self, func, dfunc):
		self.func = func
		self.dfunc = dfunc

# sigmoid = ActivationFunction((lambda x:  1 / (1 + math.exp(-x))), (lambda y:y*(1-y)))
# tanh = ActivationFunction((lambda x:(math.tanh(x))), (lambda y:1-(y*y)))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class Neural_Network:
	def __init__(self, in_nodes, hid_nodes, out_nodes):
		self.input_nodes = in_nodes
		self.hidden_nodes = hid_nodes
		self.output_nodes = out_nodes

		self.weight_ih = np.random.uniform(-1,1,(self.input_nodes, self.hidden_nodes))
		self.weight_ho = np.random.uniform(-1,1,(self.hidden_nodes, self.output_nodes))

		self.bias_h = np.random.uniform(-1,1)
		self.bias_o = np.random.uniform(-1,1)

	def setLearningRate(self, learning_rate = 0.1):
		self.learning_rate = learning_rate

	def setActivationFunction(self, func = sigmoid):
		self.activation_function = func

	def predict(self, inputs):
		inputs = np.array(inputs)
		# with tf.Session() as sess:  print(inputs.eval()) 

		hidden = np.matmul(np.reshape(inputs, (1, self.input_nodes)), self.weight_ih)
		hidden += self.bias_h
		hidden = sigmoid(hidden)


		output = np.matmul(np.reshape(hidden, (1, self.hidden_nodes)), self.weight_ho)
		output += self.bias_o
		output = sigmoid(output)
		
		result = np.argmax(output)	
		return result - 1
		
	def mutate(self, func):
		vfunc = np.vectorize(func)
		vfunc(self.weight_ih);
		vfunc(self.weight_ho);
		func(self.bias_h);
		func(self.bias_o);
		


#####################################################################


def game_start(SPEED):
    for i in range(3):
        pygame.display.set_caption("SNAKEBOT  |  Game starts in " + str(3-i) + " second(s) ...")
        pygame.time.wait(round(250 / SPEED))
        
def game_over(reason, SCORE):
	pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE) + "  |  GAME OVER. Press ESC to quit.")
	pygame.time.wait(1000)
	while True:
		event = pygame.event.wait()
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
				play_game(SCREEN_COLOR, 0)
			elif event.key == pygame.K_ESCAPE:
				break
	pygame.quit()
	sys.exit()

def menu():
	font = pygame.font.Font(None, 30)
	menu_message1 = font.render("Press enter to play, m for machine play", True, WHITE)

	gameDisplay.fill(BLACK)
	gameDisplay.blit(menu_message1, (SCREEN_SIZE/5, SCREEN_SIZE/2)) 
	pygame.display.update()

	while True:
		event = pygame.event.wait()
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
				return 1
			elif event.key == pygame.K_m:
				return 2

def populate(newSnakes, POPULATION):
	if len(newSnakes) == 0: 
		Snakes = []
		for i in range(POPULATION):
			Snakes.append(Snake())
		return Snakes
	return newSnakes
		
def chooseSnake(Snakes):
	index = 0
	r = random.uniform(0, 1)
	while r>0:
		r = r-Snakes[index].fitness
		index+= 1
	index-=1

	snake = Snakes[index]
	newSnake = (Snake(snake.Brain))
	# with tf.Session() as sess:  print(snake.Brain.weight_ih.eval()) 
	# with tf.Session() as sess:  print(newSnake.Brain.weight_ih.eval()) 
	# print()
	return newSnake

def breed(Snakes):
	newSnakes = []
	for _ in Snakes:
		snake = chooseSnake(Snakes)
		snake.Brain.mutate(snake.mutate)
		newSnakes.append(snake)
	return newSnakes



def machine_play(SCREEN_COLOR, SCORE, DISPLAY, GEN, LOG, Snakes):
	global SPEED
	BEST_ONLY = False
	deadSnakes = []
	Score = 0
	food = Food()
	FOOD = True
	clock = pygame.time.Clock()
	if DISPLAY:
		gameDisplay.fill(SCREEN_COLOR)
	game_start(SPEED)
	stats = pygame.display.set_caption("SNAKEBOT  |  Best Score: " + str(SCORE) + "  |  Generation: " + str(GEN))

	while len(Snakes) != 0:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()			
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					sys.exit()
				elif event.key == pygame.K_RIGHT:
					SPEED += 1
				elif event.key == pygame.K_LEFT:
					if SPEED != 1:
						SPEED -= 1
				elif event.key == pygame.K_b:
					if (not BEST_ONLY):
						BEST_ONLY = True
						print("Showing Best Snake Only")
					else:
						BEST_ONLY = False
						print("Showing All Snakes")
						for snake in Snakes:
							snake.DISPLAY = True

		if BEST_ONLY:
			best = max(snake.SCORE for snake in Snakes)
			for snake in Snakes:
				snake.DISPLAY = False
				if snake.SCORE == best:
					snake.DISPLAY = True
		if DISPLAY:
			clock.tick(15*SPEED)
			gameDisplay.fill(SCREEN_COLOR)
		pygame.display.set_caption("SNAKEBOT  |  Best Score: " + str(SCORE) + "  |  Generation: " + str(GEN) + "  |  Score: " + str(Score))
		for snake in Snakes:

			choice = snake.think(food)
			snake.move_decision(choice)

			snake.update()
			if snake.DISPLAY and DISPLAY:
				snake.show()
			if snake.eats(food.x, food.y):
				FOOD = False
				snake.SCORE += 1
				Score = max(snake.SCORE for snake in Snakes)
				snake.hunger=0
			if snake.terminate(LOG):
				Snakes.remove(snake)
				deadSnakes.append(snake)

		if FOOD == False:
			food = Food()
			FOOD = True
		food.update()
		if DISPLAY:
			food.show()
			pygame.display.update()

	Score = max(snake.SCORE for snake in deadSnakes)
	SCORE = max(Score, SCORE)
	if LOG == 1 or LOG == 2:
		print("SNAKEBOT  |  Best Score: " + str(SCORE) + "  |  Generation " + str(GEN) + " Best: " + str(Score))

	Sum = sum(snake.SCORE for snake in deadSnakes)
	if Sum != 0:
		for snake in deadSnakes:
			snake.fitness = snake.SCORE/Sum

	return SCORE, deadSnakes

newSnakes = []

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()			
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				sys.exit()
	GEN += 1
	Snakes = populate(newSnakes, POPULATION)
	SCORE, deadSnakes = machine_play(SCREEN_COLOR, SCORE, DISPLAY, GEN, LOG, Snakes)
	if max(snake.SCORE for snake in deadSnakes) != 0:
		newSnakes = breed(deadSnakes)


#	stats = pygame.display.set_caption("SNAKEBOT  |  Generation: " + str(GEN) + "  |  Score: " + str(SCORE))
