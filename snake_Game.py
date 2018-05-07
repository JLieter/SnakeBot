import pygame
import sys
from random import randint, randrange

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (100,100,100)
GREEN = (0,255,0)

pygame.init()

BLOCK_SIZE = 20
SCREEN_SIZE = 800
SCORE = 0
GEN = 0
SNAKE_COLOR = GREEN
TAIL_COLOR = GREEN


#######################MAKE SEPERATE FILE############################

class Snake:
	def __init__(self):
		self.x = SCREEN_SIZE/4
		self.y = SCREEN_SIZE/8
		self.x_speed = 1
		self.y_speed = 0
		self.tail_size = 0
		self.tail = []


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

	def eats(self, pos_x, pos_y):
		if self.x == pos_x and self.y == pos_y:
			self.tail_size += 1
			return True

	def check_collision(self):
        # Check if the head collides with the edges of the board
		if self.x >= (SCREEN_SIZE) or self.x < 0:
			return True
		elif self.y > (SCREEN_SIZE) or self.y < 0:
			return True
        # Check if the head collides with the body
		for body in self.tail[1:]:
			if self.x == body[0] and self.y == body[1]:
				return True
		return False

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

def game_start():
    for i in range(3):
        pygame.display.set_caption("SNAKEBOT  |  Game starts in " + str(3-i) + " second(s) ...")
        pygame.time.wait(500)
        
def game_over():
    pygame.display.set_caption("SNAKEBOT  |  Generation: " + str(GEN) + "  |  Score: " + str(SCORE) + "  |  GAME OVER. Press any key to quit ...")
    pygame.time.wait(1000)
    while True:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            break
    pygame.quit()
    sys.exit()




gameExit = True
snake = Snake()
food = Food()
clock = pygame.time.Clock()
gameDisplay = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
gameDisplay.fill(BLACK)
game_start()
stats = pygame.display.set_caption("SNAKEBOT  |  Generation: " + str(GEN) + "  |  Score: " + str(SCORE))

while gameExit:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			gameExit = False			
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				gameExit = False
			elif event.key == pygame.K_UP:
				snake.direction(0,-1)
			elif event.key == pygame.K_DOWN:
				snake.direction(0,1)
			elif event.key == pygame.K_LEFT:
				snake.direction(-1,0)
			elif event.key == pygame.K_RIGHT:
				snake.direction(1,0)
				
	clock.tick(15)
	gameDisplay.fill(BLACK)
	snake.update()
	snake.show()
	if (snake.check_collision()):
		game_over()
	if snake.eats(food.x, food.y):
		food = Food()
		#snake.grow()
		SCORE += 1
		pygame.display.set_caption("SNAKEBOT  |  Generation: " + str(GEN) + "  |  Score: " + str(SCORE))
	food.update()
	food.show()
	pygame.display.update()
