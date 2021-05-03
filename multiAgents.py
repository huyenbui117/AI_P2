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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        minScareTime = min(newScaredTimes)

        nearestGhost = min([manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates])
        if minScareTime > 0:
            nearestGhost *= -100
        elif nearestGhost < 2:
            nearestGhost = -100000
        elif nearestGhost >= 2:
            nearestGhost = 1 / nearestGhost

        foodList = newFood.asList()

        if foodList:
            nearestFood = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            nearestFood = 0

        oldPos = currentGameState.getPacmanPosition()
        if oldPos == newPos: return -9999

        foodLeft_weight = -100
        distanceToGhost_weight = 1
        nearestFood_weight = -2

        return nearestGhost * distanceToGhost_weight + len(
            foodList) * foodLeft_weight + nearestFood * nearestFood_weight


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"
        result = self.minimax(gameState.generateSuccessor(0, gameState.getLegalActions(0)[0]), 1, 1)
        for i in gameState.getLegalActions(0):
            if self.minimax(gameState.generateSuccessor(0, i), 1, 1) > result:
                result = self.minimax(gameState.generateSuccessor(0, i), 1, 1)
        for i in gameState.getLegalActions(0):
            if self.minimax(gameState.generateSuccessor(0, i), 1, 1) == result:
                return i

    def minimax(self, state, agentIndex, depth):
        if agentIndex == state.getNumAgents():
            if depth == self.depth:
                return self.evaluationFunction(state)
            return self.minimax(state, 0, depth + 1)

        cases = state.getLegalActions(agentIndex)
        if len(cases) == 0: return self.evaluationFunction(state)

        future = (self.minimax(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in cases)

        if agentIndex == 0:
            return max(future)
        return min(future)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maX(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def maX(self, gameState, agent, depth, alpha, beta):
        best = ("", -float("inf"))

        for i in gameState.getLegalActions(agent):
            action = (i, self.AB(gameState.generateSuccessor(agent, i), (depth + 1) % gameState.getNumAgents(),
                                 depth + 1, alpha, beta))
            best = max(best, action, key=lambda x: x[1])

            if best[1] > beta:
                return best
            alpha = max(alpha, best[1])

        return best

    def miN(self, gameState, agent, depth, alpha, beta):
        best = ("", float("inf"))

        for i in gameState.getLegalActions(agent):
            action = (i, self.AB(gameState.generateSuccessor(agent, i),
                                 (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            best = min(best, action, key=lambda x: x[1])

            if best[1] < alpha:
                return best
            beta = min(beta, best[1])

        return best

    def AB(self, gameState, agent, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() \
                or depth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        if agent == 0:
            return self.maX(gameState, agent, depth, alpha, beta)[1]
        return self.miN(gameState, agent, depth, alpha, beta)[1]


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

        numOfAgent = gameState.getNumAgents()
        # For each possible direction, add its prediction score to this array
        ActionScore = []

        def expectimax(currentState, treeDeep):
          # Some special case (leaf of the tree, already won or loss)
          if treeDeep >= self.depth*numOfAgent or currentState.isWin() or currentState.isLose():
            ans = self.evaluationFunction(currentState)
            return ans

          # Pacman Max  
          if treeDeep%numOfAgent == 0: 
            # In case there are no legal actions for Pacman, return -1e4 (losing state)
            result = -1e4

            # Else, find the way with the maximum possible score
            for a in currentState.getLegalActions(treeDeep%numOfAgent):
              # Stop is always not a good choice
              if a == 'Stop':
                continue
              # Generate next Game State
              sdot = currentState.generateSuccessor(treeDeep%numOfAgent,a)
              # Recursive
              result = max(result, expectimax(sdot, treeDeep+1))
              # Score of each branch of the first node
              if treeDeep == 0:
                ActionScore.append(result)
            
            return result

          # Random move of a ghost
          else:
            # An array of Score with each possible Ghost's direction
            successorScore = [expectimax(currentState.generateSuccessor(treeDeep%numOfAgent,a), treeDeep+1) for a in currentState.getLegalActions(treeDeep%numOfAgent)]

            # The average (or expected) score of the array above
            return sum([ float(x)/len(successorScore) for x in successorScore])

        result = expectimax(gameState, 0)
        # [x for x in gameState.getLegalActions(0) if x != 'Stop']: Array of direction (Left, Right, Up, Down)
        # [ActionScore.index(max(ActionScore))]                   : Index with the corresponding highest score 
        return [x for x in gameState.getLegalActions(0) if x != 'Stop'][ActionScore.index(max(ActionScore))]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    result = currentGameState.getScore()
    playerPos = currentGameState.getPacmanPosition()

    #1: Score from Ghosts
    for ghost in currentGameState.getGhostStates():
      disGhost = manhattanDistance(playerPos, ghost.getPosition())
      if ghost.scaredTimer > 0:
        result += pow(max(8 - disGhost, 0), 2)
      else:
        result -= pow(max(7 - disGhost, 0), 2)

    #2: Score from Foods
    if len(currentGameState.getFood().asList()) > 0:
      result += 1.0/min(manhattanDistance(playerPos, food) for food in currentGameState.getFood().asList())

    #3: Score from Capsules
    if len(currentGameState.getCapsules()) > 0:
      result += 50.0/min(manhattanDistance(playerPos, Cap) for Cap in currentGameState.getCapsules())
    
    return result


# Abbreviation
better = betterEvaluationFunction
