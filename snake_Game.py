import pygame
import sys
from random import randint, randrange
import time
import math
import numpy as np


BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (100,100,100)
GREEN = (0,255,0)
WHITE = (255,255,255)



BLOCK_SIZE = 20
SCREEN_SIZE = 800
SCORE = 0
GEN = 0
SNAKE_COLOR = GREEN
TAIL_COLOR = GREEN
SCREEN_COLOR = BLACK
DISPLAY = True

pygame.init()
gameDisplay = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))


#######################MAKE SEPERATE FILE############################

class Snake:
	def __init__(self):
		self.x = SCREEN_SIZE/4
		self.y = SCREEN_SIZE/8
		self.x_speed = 1
		self.y_speed = 0
		self.tail_size = 0
		self.tail = []
		self.STARVATION_RATE = 25
		self.Direction = "RIGHT"

	def update(self):
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
		if y == -1:
			self.Direction =  "DOWN"

	def eats(self, pos_x, pos_y):
		if self.x == pos_x and self.y == pos_y:
			self.tail_size += 1
			return True

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

	def starved(self, mark):
		if (time.clock() - mark) > self.STARVATION_RATE:
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

	def check_obstacle_left(self):
		if self.Direction == "RIGHT":
			one_step = self.y - BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (self.x, one_step) in self.tail:
				return 0
		elif self.Direction == "LEFT":
			one_step = self.y + BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (self.x, one_step) in self.tail:
				return 0
		elif self.Direction == "UP":
			one_step = self.y - BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (one_step, self.y) in self.tail:
				return 0
		elif self.Direction == "DOWN":
			one_step = self.y + BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (one_step, self.y) in self.tail:
				return 0
		return 1

	def check_obstacle_right(self):
		if self.Direction == "RIGHT":
			one_step = self.y + BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (self.x, one_step) in self.tail:
				return 0
		elif self.Direction == "LEFT":
			one_step = self.y - BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (self.x, one_step) in self.tail:
				return 0
		elif self.Direction == "UP":
			one_step = self.y + BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (one_step, self.y) in self.tail:
				return 0
		elif self.Direction == "DOWN":
			one_step = self.y - BLOCK_SIZE
			if one_step == 0 or one_step == SCREEN_SIZE or (one_step, self.y) in self.tail:
				return 0
		return -1



#####################################################################

class Food:
	def __init__(self):
		self.x = (randint(0,((SCREEN_SIZE-BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE)
		self.y = (randint(0,((SCREEN_SIZE-BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE)

	def update(self):
		self.COLOR = (randrange(254)|64, randrange(254)|64, randrange(254)|64)

	def show(self):
		pygame.draw.rect(gameDisplay, self.COLOR, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])

#####################################################################


class Neural_Network:
	def __init__(self):
		self.score = 0

	def think(self, input1, input2, input3):
		input1 = input1 / 3
		input2 = input2 / 3
		input3 = input3 / 3
		return input1 + input2 + input3


	def predict(self, inVal = None):
		if inVal != None:
			if inVal >= -1 and inVal <= -0.33:
				output = -1
			elif inVal <= 1 and inVal >= 0.33:
				output = 1
			else:
				output = 0
				output = (randint(-4,4))
				if output == -4:
					return -1
				elif output == 4: 
					return 1
				else: 
					return 0
		return output


#####################################################################


def game_start():
    for i in range(3):
        pygame.display.set_caption("SNAKEBOT  |  Game starts in " + str(3-i) + " second(s) ...")
        pygame.time.wait(250)
        
def game_over(reason, SCORE):
#	if (reason == "Collision"):
	pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE) + "  |  GAME OVER: "+reason+". Press ENTER to restart or ESC to quit.")
#	elif (reason == "Starvation"):
 #		pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE) + "  |  GAME OVER: SNAKE STARVED. Press ENTER to restart or ESC to quit.")
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


def play_game(SCREEN_COLOR, SCORE):
	snake = Snake()
	food = Food()
	clock = pygame.time.Clock()
	gameDisplay.fill(SCREEN_COLOR)
	game_start()
	stats = pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE))

	mark = time.clock()
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()			
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					sys.exit()
				elif event.key == pygame.K_UP:
					snake.direction(0,-1)
				elif event.key == pygame.K_DOWN:
					snake.direction(0,1)
				elif event.key == pygame.K_LEFT:
					snake.direction(-1,0)
				elif event.key == pygame.K_RIGHT:
					snake.direction(1,0)
					
		clock.tick(15)
		gameDisplay.fill(SCREEN_COLOR)
		snake.update()
		snake.show()
		if (snake.starved(mark)):
			game_over("STARVATION", SCORE)
		if (snake.check_collision()):
			game_over("COLLISION", SCORE)
		if snake.eats(food.x, food.y):
			food = Food()
			SCORE += 1
			pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE))
			mark = time.clock()
		food.update()
		food.show()
		pygame.display.update()

def machine_play(SCREEN_COLOR, SCORE, DISPLAY):
	Brain = Neural_Network()
	snake = Snake()
	food = Food()
	clock = pygame.time.Clock()
	if DISPLAY:
		gameDisplay.fill(SCREEN_COLOR)
	game_start()
	stats = pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE))

	mark = time.clock()
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()			
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					sys.exit()

		input1 = snake.check_obstacle()
		input2 = snake.check_obstacle_left()
		input3 = snake.check_obstacle_right()
		print (input1, input2, input3)

		value = Brain.think(input1, input2, input3)
		keystroke = Brain.predict(value)
		print (keystroke)
		if snake.Direction == "LEFT":
			if keystroke == -1:
				snake.direction(0, -1)
			if keystroke == 1:
				snake.direction(0, 1)
		if snake.Direction == "RIGHT":
			if keystroke == -1:
				snake.direction(0, 1)
			if keystroke == 1:
				snake.direction(0, -1)
		if snake.Direction == "UP":
			if keystroke == -1:
				snake.direction(-1, 0)
			if keystroke == 1:
				snake.direction(1, 0)
		if snake.Direction == "DOWN":
			if keystroke == -1:
				snake.direction(1, 0)
			if keystroke == 1:
				snake.direction(-1, 0)

		if DISPLAY:
			clock.tick(15)
			gameDisplay.fill(SCREEN_COLOR)
		snake.update()
		if DISPLAY:
			snake.show()
		if (snake.starved(mark)):
			reason == "STARVATION"
			pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE) + "  |  GAME OVER: "+reason+". Press ENTER to restart or ESC to quit.")
			pygame.time.wait(1000)
			machine_play(SCREEN_COLOR, SCORE, True)
		if (snake.check_collision()):
			reason = "COLLISION"
			pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE) + "  |  GAME OVER: "+reason+". Press ENTER to restart or ESC to quit.")
			pygame.time.wait(1000)
			machine_play(SCREEN_COLOR, SCORE, True)
		if snake.eats(food.x, food.y):
			food = Food()
			SCORE += 1
			pygame.display.set_caption("SNAKEBOT  |  Score: " + str(SCORE))
			mark = time.clock()
		food.update()
		if DISPLAY:
			food.show()
			pygame.display.update()



choice = menu()
if choice == 1:
	play_game(SCREEN_COLOR, SCORE)
elif choice == 2:
	machine_play(SCREEN_COLOR, SCORE, DISPLAY)



#	stats = pygame.display.set_caption("SNAKEBOT  |  Generation: " + str(GEN) + "  |  Score: " + str(SCORE))
