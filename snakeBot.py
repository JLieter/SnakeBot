import pygame
import sys
from random import randint, randrange, gauss, getrandbits
import time
import math
import numpy as np
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
MAX_STARVATION_RATE = 500
MUTATION_RATE = 0.01
WEIGHT_MUTATION_RATE = 0.8
CONNECTION_MUTATION_RATE = 0.1
NODE_MUTATION_RATE = 0.1
BLOCK_SIZE = 20
SCREEN_SIZE = 800
GEN = 0
BEST_ONLY = False
COLORED_FIT = False
SNAKE_COLOR = GREEN
TAIL_COLOR = GREY
FOOD_COLOR = "Random"
# FOOD_COLOR = RED
SCREEN_COLOR = BLACK
FOOD = False
CompatibilityDistanceThreshold = 1
C1 = 1
C2 = 1
C3 = 0.4

connectionHistory = []
SpeciesList = []
Genomes = []


pygame.init()
if DISPLAY:
	gameDisplay = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))


####################### Snake Class ############################

class Snake:
	def __init__(self, Brain):
		self.DISPLAY = True
		self.x = SCREEN_SIZE/4
		self.y = SCREEN_SIZE/8
		self.x_speed = 1
		self.y_speed = 0
		self.tail_size = 0
		self.tail = []
		self.Direction = "RIGHT"
		self.SCORE = 0
		self.lifetime = 0
		self.fitness = 0
		self.hunger = STARVATION_RATE
		self.Brain = Brain
		self.Brain.generateNetwork()
			
	def think(self, food):
		inputs = self.look(food)
		result = self.Brain.feedForward(inputs)
		# print(result)
		pred = result.index(max(result))
		return pred + 1

	def update(self):
		self.hunger -= 1
		self.lifetime += 1
		self.calcFitness()
		self.tail.insert(0, (self.x, self.y))
		if self.tail_size != len(self.tail):
			self.tail.pop()

		self.x += (self.x_speed*BLOCK_SIZE)
		self.y += (self.y_speed*BLOCK_SIZE)

	def show(self):
		for i in range(self.tail_size):
			pygame.draw.rect(gameDisplay, TAIL_COLOR, [self.tail[i][0], self.tail[i][1], BLOCK_SIZE, BLOCK_SIZE])

		if not COLORED_FIT:
			pygame.draw.rect(gameDisplay, SNAKE_COLOR, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])
		else:
			snake_color = ((self.fitness // 255*self.SCORE if self.fitness // 255*self.SCORE < 255 else 255), 0, 255)
			pygame.draw.rect(gameDisplay, snake_color, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])

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
			if keystroke == 1:
					self.direction(1, 0)
			elif keystroke == 2:
				self.direction(-1, 0)
			elif keystroke == 3:
				self.direction(0, 1)
			elif keystroke == 4:
				self.direction(0, -1)
			else:
				pritn("ERROR")
			

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
		if self.hunger == 0:
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

	def look(self, food):
		arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		arr[0],arr[1],arr[2] = self.lookDir(0, 1, food)
		arr[3],arr[4],arr[5] = self.lookDir(1, 1, food)
		arr[6],arr[7],arr[8] = self.lookDir(1, 0, food)
		arr[9],arr[10],arr[11] = self.lookDir(1, -1, food)
		arr[12],arr[13],arr[14] = self.lookDir(0, -1, food)
		arr[15],arr[16],arr[17] = self.lookDir(-1, -1, food)
		arr[18],arr[19],arr[20] = self.lookDir(-1, 0, food)
		arr[21],arr[22],arr[23] = self.lookDir(-1, 1, food)
		return arr


	def lookDir(self, x, y, food):
		stats = [0,0,0]
		distance = 0
		pos_x = self.x
		pos_y = self.y
		while (pos_x > 0 and pos_x < SCREEN_SIZE and pos_y > 0 and pos_y < SCREEN_SIZE):
			distance+=1
			pos_x += ((distance*x) * BLOCK_SIZE)
			pos_y += ((distance*y) * BLOCK_SIZE)
			if (food.x == pos_x and food.y == pos_y):
				stats[0] = 1
			if ((pos_x,pos_y) in self.tail):
				stats[2] = distance
		if distance != 0:
			stats[1] = 1 / distance
		return stats

	def calcFitness(self):
		if(self.SCORE < 10):
			self.fitness = math.floor(self.lifetime * self.lifetime) * pow(2,self.SCORE)
		else:
			self.fitness = math.floor(self.lifetime * lifetime)
			self.fitness *= pow(2,10)
			self.fitness *= (score-9)


######################## Food Class ############################

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

####################### Brain Class ############################

class ActivationFunction:
	def __init__(self, func, dfunc):
		self.func = func
		self.dfunc = dfunc


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def mutate(x):
	if (np.random.random() < MUTATION_RATE):
		return np.random.uniform(-1,1)
	else:
		offset = gauss(0,1) / 50
		newx = x + offset
		if newx > 1:
			newx = 1
		if newx < -1:
			newx = -1
		return newx


class Genome:
	def __init__(self, inputs, outputs):
		self.connections = []
		self.nodes = []
		self.network = []
		self.fitness = 0
		self.adjustedFitness = 0
		self.inputs = inputs
		self.outputs = outputs
		self.nextNode = 0
		self.biasNode = 0

		for i in range(inputs):
			self.addNode(self.nextNode, "Input")

		for i in range(outputs):
			self.addNode(self.nextNode, "Output")
		
		self.biasNode = self.nextNode
		self.addNode(self.nextNode, "Input")


	def addNode(self, ID, TYPE):
		self.nodes.append(NodeGene(ID, TYPE))
		self.nextNode += 1


	def connectNodes(self):
		for connection in self.connections:
			if connection not in connection.inNode.outputConnections:
				connection.inNode.outputConnections.append(connection)

		for node in self.nodes:
			for conn in node.outputConnections:
				if conn not in self.connections:
					node.outputConnections.remove(conn)



	def generateNetwork(self):
		self.network = []
		self.connectNodes()

		for layer in ["Input", "Hidden", "Output"]:
			for node in self.nodes:
				if node.TYPE == layer:
					self.network.append(node)
	


	def feedForward(self, inputs):

		for i in range(len(inputs)):
			self.network[i].outputValue = inputs[i]

		self.nodes[self.biasNode].outputValue = 1


		for node in self.network:
			node.engage()

		outputs = []
		for node in self.nodes:
			if node.TYPE == "Output":
				outputs.append(node.outputValue)

		for node in self.nodes:
			node.inputSum = 0
			for connection in node.outputConnections:
				connection.outNode.inputSum = 0
				connection.outNode.outputValue = 0

		return outputs

	def mutation(self):
		for connection in self.connections:
			connection.weight = mutate(connection.weight)


	def addConnectionMutation(self):
		#If there are at least 2 nodes select 2 random unique nodes
		node1 = self.nodes[randint(0, len(self.nodes)-1)]
		node2 = self.nodes[randint(0, len(self.nodes)-1)]
		while (node1.ID == node2.ID or node1.TYPE == node2.TYPE):
			node2 = self.nodes[randint(0, len(self.nodes)-1)]
		
		#Flip Node1 and Node2 if they are back propagating
		if (node1.TYPE == "Hidden" and node2.TYPE == "Input"):
			node1, node2 = node2, node1
		elif (node1.TYPE == "Hidden" and node2.TYPE == "Output"):
			node1, node2 = node2, node1
		elif (node1.TYPE == "Output" and node2.TYPE == "Input"):
			node1, node2 = node2, node1

		#Check if Connection Exists
		for connection in self.connections:
			if (connection.inNode == node1 and connection.outNode == node2):
				return

		#Add Connection
		weight = np.random.uniform(-1,1)
		self.connections.append(ConnectionGene(node1, node2, weight))


	def addNodeMutation(self):
		connection = self.connections[randint(0, len(self.connections)-1)]
		connection.expressed = False

		self.addNode(self.nextNode, "Hidden")
		node = self.nodes[self.nextNode-1]
		

		weight = np.random.uniform(-1,1)
		self.connections.append(ConnectionGene(connection.inNode, node, weight))
		self.connections.append(ConnectionGene(node, connection.outNode, connection.weight))


	def crossover(self, parent2):
		child = Genome(self.inputs, self.outputs)
		child.nodes = []
		child.connections = []
		child.network = []
		child.fitness = 0
		child.adjustedFitness = 0
		child.nextNode = self.nextNode
		child.biasNode = self.biasNode

		for node in self.nodes:
			child.nodes.append(node.copy())

		for connection in self.connections:
			if (any(conn for conn in parent2.connections if connection.Innovation == conn.Innovation)):
				if bool(getrandbits(1)):
					childConGene = connection.copy()
				else:
					for conn in parent2.connections:
						if conn.Innovation == connection.Innovation:
							childConGene = conn.copy()
				child.connections.append(childConGene)
			else:
				childConGene = connection.copy()
				child.connections.append(childConGene)	 


		return child



class ConnectionGene:
	def __init__(self, inNode, outNode, weight):
		self.inNode = inNode
		self.outNode = outNode
		self.weight = weight
		self.expressed = True
		self.Innovation = 0
		self.getInnovation()


	def copy(self):
		connection = ConnectionGene(self.inNode, self.outNode, self.weight)
		connection.expressed = self.expressed
		return connection

	def getInnovation(self, connectionHistory = connectionHistory):
		for connection in connectionHistory:
			if connection.inNode.ID == self.inNode.ID:
				if connection.outNode.ID == self.outNode.ID:
					self.Innovation = connection.Innovation
					return
		self.Innovation = len(connectionHistory)
		connectionHistory.append(self)
					


class NodeGene:
	def __init__(self, ID, TYPE):
		self.ID = ID
		self.TYPE = TYPE
		self.inputSum = 0 
		self.outputValue = 0
		self.SET = False
		self.outputConnections = []

	def engage(self):
		if self.TYPE != "Input":
			self.outputValue = sigmoid(self.inputSum)
		for connection in self.outputConnections:
			if connection.expressed:
				connection.outNode.inputSum += self.outputValue * connection.weight


	def copy(self):
		node = NodeGene(self.ID, self.TYPE)
		node.outputConnections = self.outputConnections
		return node


class Species():
	def __init__(self, mascot):
		self.mascot = mascot
		self.members = []
		self.memFitness = {}
		self.adjustedFitness = 0
		self.members.append(mascot)

	def clearSpecies(self):
		self.mascot = self.members[random.randint(len(self.members))]
		self.members.clear()
		self.memFitness.clear()
		self.adjustedFitness = 0

	def calcFitness(self):
		for genome in self.members:
			genome.adjustedFitness = genome.fitness / len(self.members)
			self.memFitness[genome] = genome.adjustedFitness
			self.adjustedFitness += genome.adjustedFitness


def countGenes(genome1, genome2):
	matchingGenes = 0
	disjointGenes = 0
	excessGenes = 0
	weightDifference = 0
	averageWeightDifference = 0

	highestInnovation1 = max(connection.Innovation for connection in genome1.connections)
	highestInnovation2 = max(connection.Innovation for connection in genome2.connections)
	index = max(highestInnovation1, highestInnovation2)

	for i in range(index+1):
		#If they are matching genes
		if ( (sum(connection.Innovation == i for connection in genome1.connections) != 0) and (sum(connection.Innovation == i for connection in genome2.connections) != 0)):
			matchingGenes += 1
			weightDifference += abs( next(connection.weight for connection in genome1.connections if connection.Innovation == i) - next(connection.weight for connection in genome2.connections if connection.Innovation == i))
		#If genome1 contains this gene, but genome2 does not
		elif ( (sum(connection.Innovation == i for connection in genome1.connections) != 0) and (sum(connection.Innovation == i for connection in genome2.connections) == 0)):
			if highestInnovation2 > i:
				disjointGenes += 1
			elif highestInnovation2 < i:
				excessGenes += 1
		#If genome2 contains this gene, but genome1 does not
		elif ( (sum(connection.Innovation == i for connection in genome1.connections) == 0) and (sum(connection.Innovation == i for connection in genome2.connections) != 0)):
			if highestInnovation1 > i:
				disjointGenes += 1
			elif highestInnovation1 < i:
				excessGenes += 1			

	if matchingGenes > 0:
		averageWeightDifference = weightDifference / matchingGenes
	return matchingGenes, disjointGenes, excessGenes, averageWeightDifference


def CompatibilityDistance(genome1, genome2, c1, c2, c3):
	matchingGenes, disjointGenes, excessGenes, averageWeightDifference = countGenes(genome1, genome2)
	N = max( len(genome1.nodes), len(genome2.nodes) )
	N -= 20
	if N < 1:
		N = 1

	return ( ((excessGenes * c1)/N) + ((disjointGenes * c2)/N) + (averageWeightDifference * c3) )


class Population:
	def __init__(self, POPULATION = POPULATION):
		global Genomes
		self.POPULATION = POPULATION
		self.members = []
		self.initialGenomes()

	def initialGenomes(self):
		for i in range(self.POPULATION):
			genome = Genome(24,4)
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			genome.addConnectionMutation()
			Genomes.append(genome)

	def populate(self, Genomes):
		for genome in Genomes:
			self.members.append(Snake(genome))
		return self.members


####################### Game Code ##############################


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


def Selection(Genomes):
	#Method for randomly selecting a candidate in a list with increasing percentage based on its fitness
	Sum = sum(genome.fitness for genome in Genomes)
	r = np.random.uniform(0,Sum)
	runningSum = 0

	for index in range(len(Genomes)):
		runningSum += Genomes[index].fitness
		if runningSum > r:
			return Genomes[index]

	return Genomes[0]


def Speciate(Genomes, SpeciesList = SpeciesList, POPULATION = POPULATION):
	nextGenGenomes = []

	#Select New Species mascot for each species
	for species in SpeciesList:
		bestG = max(genome.fitness for genome in species.members)
		for genome in species.members:
			if genome.fitness == bestG:
				species.mascot = genome
		species.members = []
		species.members.append(species.mascot)
		species.memFitness.clear()

	#Categorize each Genome into a species
	for genome in Genomes:
		foundSpecies = False
		for species in SpeciesList:
			if ( CompatibilityDistance(genome, species.mascot, C1, C2, C3) < CompatibilityDistanceThreshold ):
				species.members.append(genome)
				foundSpecies = True
				break
		if not foundSpecies:
			SpeciesList.append(Species(genome))

	#Calculate the fitness of each Species and find best genome per species
	for species in SpeciesList:
		species.calcFitness()
		topGenome = max(species.memFitness, key=species.memFitness.get)
		nextGenGenomes.append(topGenome)

	#Establish Base population size again with Crossovers
	while len(nextGenGenomes) < POPULATION:
		p1 = Selection(Genomes)
		p2 = Selection(Genomes)
		while (p1 == p2):
			p2 = Selection(Genomes)
		
		if (p1.fitness >= p2.fitness):
			child = p1.crossover(p2)
		else:
			child = p2.crossover(p1)

		if (np.random.random() < WEIGHT_MUTATION_RATE):
			child.mutation()

		if (np.random.random() < CONNECTION_MUTATION_RATE):
			child.addConnectionMutation()

		if (np.random.random() < NODE_MUTATION_RATE):
			child.addNodeMutation()

		nextGenGenomes.append(child)

	return nextGenGenomes


def machine_play(SCREEN_COLOR, SCORE, DISPLAY, GEN, LOG, Snakes):
	global SPEED
	global BEST_ONLY
	global COLORED_FIT
	global FOOD_COLOR
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
				elif event.key == pygame.K_c:
					if not COLORED_FIT:
						COLORED_FIT = True
						print("Colored Fitness Mode")
					else:
						COLORED_FIT = False
						print("Normal Coloring Mode")
				elif event.key == pygame.K_f:
					if FOOD_COLOR == RED:
						FOOD_COLOR = "Random"
						print("Colored Food")
					else:
						FOOD_COLOR = RED
						print("Red Food")
				elif event.key == pygame.K_1:
					if SPEED != 1:
						print("Speed set to 1")
						SPEED = 1
				elif event.key == pygame.K_2:
					if SPEED != 2:
						print("Speed set to 2")
						SPEED = 2
				elif event.key == pygame.K_3:
					if SPEED != 3:
						print("Speed set to 3")
						SPEED = 3
				elif event.key == pygame.K_4:
					if SPEED != 4:
						print("Speed set to 4")
						SPEED = 4
				elif event.key == pygame.K_5:
					if SPEED != 5:
						print("Speed set to 5")
						SPEED = 5
				elif event.key == pygame.K_9:
					if SPEED != 9:
						print("Speed set to 9")
						SPEED = 9

		if BEST_ONLY:
			best = max(snake.Brain.fitness for snake in Snakes)
			for snake in Snakes:
				snake.DISPLAY = False
				if snake.Brain.fitness == best:
					snake.DISPLAY = True
		if DISPLAY:
			clock.tick(15*SPEED)
			gameDisplay.fill(SCREEN_COLOR)
		pygame.display.set_caption("SNAKEBOT  |  Best Score: " + str(SCORE) + "  |  Generation: " + str(GEN) + "  |  Score: " + str(Score))
		for snake in Snakes:
			if snake.terminate(LOG):
				Snakes.remove(snake)
				snake.Brain.fitness = snake.fitness
				deadSnakes.append(snake)
				continue		
			choice = snake.think(food)
			snake.move_decision(choice)

			snake.update()
			if snake.DISPLAY and DISPLAY:
				snake.show()
			if snake.eats(food.x, food.y):
				FOOD = False
				snake.SCORE += 1
				Score = max(snake.SCORE for snake in Snakes)
				if snake.hunger < (MAX_STARVATION_RATE-100):
					snake.hunger+=100

		if FOOD == False:
			food = Food()
			FOOD = True
		food.update()
		if DISPLAY:
			food.show()
			pygame.display.update()
	Score = max(snake.SCORE for snake in deadSnakes)
	Fitness = max(snake.Brain.fitness for snake in deadSnakes)
	SCORE = max(Score, SCORE)
	if LOG == 1 or LOG == 2:
		print("SNAKEBOT  |  Best Score: " + str(SCORE) + "  |  Generation " + str(GEN) + " Best Score: " + str(Score) + ", Fitness: " + str(Fitness))

	return SCORE


####################### Event Loop #############################

POP = Population()

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()			
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				sys.exit()
	GEN += 1
	Snakes = POP.populate(Genomes)
	SCORE = machine_play(SCREEN_COLOR, SCORE, DISPLAY, GEN, LOG, Snakes)
	Genomes = Speciate(Genomes)


