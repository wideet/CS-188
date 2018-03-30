# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return 1000000
        nearestGhostPosition = currentGameState.getGhostPosition(1)
        distanceFromGhost = util.manhattanDistance(nearestGhostPosition, newPos)
        currScore = distanceFromGhost + successorGameState.getScore()
        foodList = newFood.asList()
        nearestFood = 10000000
        for food in foodList:
            distanceToFood = util.manhattanDistance(food, newPos)
            if (distanceToFood < nearestFood):
                nearestFood = distanceToFood
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            currScore += 100
        if action == Directions.STOP:
            currScore -= 100
        currScore -= 3 * nearestFood
        powerPelletList = currentGameState.getCapsules()
        if successorGameState.getPacmanPosition() in powerPelletList:
            currScore += 120
        return currScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        depth = self.depth
        moves = gameState.getLegalActions(0)
        first = True
        maxScore = 0
        maxMove = None
        for move in moves:
            score = self.helper(gameState.generateSuccessor(0, move), 1, depth)
            if first:
                first = False
                maxScore = score
                maxMove = move
            else:
                if score > maxScore:
                    maxScore = score
                    maxMove = move
        return maxMove

    def helper(self, gameState, agent, depth):
        if depth > 0 and not gameState.isWin() and not gameState.isLose():
            if agent < gameState.getNumAgents():
                actions = gameState.getLegalActions(agent)
                scores = []
                for action in actions:
                    scores.append(self.helper(gameState.generateSuccessor(agent, action), agent + 1, depth))
                if agent > 0:
                    return min(scores)
                else:
                    return max(scores)
            else:
                return self.helper(gameState, 0, depth - 1)
        else:
            return self.evaluationFunction(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # depth = self.depth
        # moves = gameState.getLegalActions(0)
        # first = True
        # maxScore = 0
        # maxMove = None
        # for move in moves:
        #     score = self.helper(gameState.generateSuccessor(0, move), 1, depth, -sys.maxsize, sys.maxsize)
        #     if first:
        #         first = False
        #         maxScore = score
        #         maxMove = move
        #     else:
        #         if score > maxScore:
        #             maxScore = score
        #             maxMove = move
        # return maxMove

        depth = self.depth
        a = -sys.maxsize
        actions = gameState.getLegalActions(0)
        maxScore = -sys.maxsize
        maxAction = None
        for action in actions:
            score = self.helper(gameState.generateSuccessor(0, action), 1, depth, a, sys.maxsize)
            if score > maxScore:
                maxScore = score
                maxAction = action
            a = max(a, maxScore)
        return maxAction

    def helper(self, gameState, agent, depth, a, b):
        if depth > 0 and not gameState.isWin() and not gameState.isLose():
            if agent < gameState.getNumAgents():
                actions = gameState.getLegalActions(agent)

                if agent == 0:
                    maxScore = -sys.maxsize
                    for action in actions:
                        score = self.helper(gameState.generateSuccessor(agent, action), agent + 1, depth, a, b)
                        maxScore = max(maxScore, score)
                        if maxScore > b:
                            return maxScore
                        a = max(a, maxScore)
                    return maxScore

                else:
                    minScore = sys.maxsize
                    for action in actions:
                        score = self.helper(gameState.generateSuccessor(agent, action), agent + 1, depth, a, b)
                        minScore = min(minScore, score)
                        if minScore < a:
                            return minScore
                        b = min(b, minScore)
                    return minScore

            else:
                return self.helper(gameState, 0, depth - 1, a, b)
        else:
            return self.evaluationFunction(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        moves = gameState.getLegalActions(0)
        first = True
        maxScore = 0
        maxMove = None
        for move in moves:
            score = self.helper(gameState.generateSuccessor(0, move), 1, depth)
            if first:
                first = False
                maxScore = score
                maxMove = move
            else:
                if score > maxScore:
                    maxScore = score
                    maxMove = move
        return maxMove

    def helper(self, gameState, agent, depth):
        if depth > 0 and not gameState.isWin() and not gameState.isLose():
            if agent < gameState.getNumAgents():
                actions = gameState.getLegalActions(agent)
                scores = []
                for action in actions:
                    scores.append(self.helper(gameState.generateSuccessor(agent, action), agent + 1, depth))
                if agent > 0:
                    return sum(scores) / float(len(scores))
                else:
                    return max(scores)
            else:
                return self.helper(gameState, 0, depth - 1)
        else:
            return self.evaluationFunction(gameState)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    if currentGameState.isWin():
        return 1000000
    if currentGameState.isLose():
    	return -1000000
    # nearestGhostPosition = currentGameState.getGhostPosition(1)
    # distanceFromGhost = util.manhattanDistance(nearestGhostPosition, newPos)
    currScore = scoreEvaluationFunction(currentGameState)
    foodList = newFood.asList()
    nearestFood = 10000000
    for food in foodList:
        distanceToFood = util.manhattanDistance(food, newPos)
        if (distanceToFood < nearestFood):
            nearestFood = distanceToFood
    numGhosts = currentGameState.getNumAgents() - 1
    distanceFromGhost = float("inf")
    for x in range(1, numGhosts + 1):
    	nextDist = util.manhattanDistance(newPos, currentGameState.getGhostPosition(x))
    	distanceFromGhost = min(distanceFromGhost, nextDist)
    currScore += 4
    currScore -= nearestFood * 1.5
    # if currentGameState.getNumFood() > successorGameState.getNumFood():
    #     currScore += 100
    # if action == Directions.STOP:
    #     currScore -= 100
    # currScore -= 3 * nearestFood
    powerPelletList = currentGameState.getCapsules()
    currScore -= 4 * len(foodList)
    currScore -= 3.5 * len(powerPelletList)
    # if currentGameState.getPacmanPosition() in powerPelletList:
    #     currScore += 120
    return currScore

# Abbreviation
better = betterEvaluationFunction

