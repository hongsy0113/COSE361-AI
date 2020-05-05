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
        # 팩맨의 움직임 이후의 다음 게임 상태
        # generatePacmanSuccessor(action) : Generates the successor state after the specified pacman move
        newPos = successorGameState.getPacmanPosition()
        # 다음 게임 상태에서 팩맨의 위치 정보
        # position 은 좌표 형식 [x,y]
        newFood = successorGameState.getFood()
        # 다음 게임 상태에서 음식의 위치 정보
        # 좌표들의 리스트
        """
              Returns a Grid of boolean food indicator variables.

              Grids can be accessed via list notation, so to check
              if there is food at (x,y), just call

              currentFood = state.getFood()
              if currentFood[x][y] == True: ...
              """
        newGhostStates = successorGameState.getGhostStates()
        # 다음 게임 상태에서 고스트들의 상태 정보
        # return self.data.agentStates[1:]
        # agentstates[0] 은 팩맨 상태, agentstates[1:] 은 고스트들 의 상태 정보


        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # 다음 게임 상태에서 각각의 고스트 들의 scared timer 값을 갖는 리스트

        "*** YOUR CODE HERE ***"

        newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]
        # newGhostPos = succesorGameState.getGhostPositions()
        # 다음 게임 상태에서 고스트들의 위치 정보

        nowPos = currentGameState.getPacmanPosition()
        # 현재 게임 상태에서 팩맨의 위치 정보
        nowFood = currentGameState.getFood()
        # 현재 게임 상태에서 음식의 위치 정보
        nowGhostStates = currentGameState.getGhostStates()
        # 현재 게임 상태에서 고스트들의 상태 정보

        score = 0

        x_p = newPos[0]
        y_p = newPos[1]
        # 다음 게임 상태에서 팩맨의 위치의 x, y 좌표
        if nowFood[x_p][y_p] == True:
            score += 10

        distancetoFood = float("inf")
        for food in newFood.asList():
            temp = manhattanDistance(newPos, food)
            if (temp<distancetoFood):
                distancetoFood = temp
        # 팩맨과 음식의 최소 맨해튼 거리 구하기

        distancetoGhost = float("inf")
        ghostindex = 0
        for ghost in newGhostPos:
            temp = manhattanDistance(newPos, ghost)
            if (temp<distancetoGhost):
                distancetoGhost = temp
                ghostindex = newGhostPos.index(ghost)
        # 팩맨과 고스트의 최소 거리 구하고 그때의 고스트 인덱스 넘버 구하기

        if newScaredTimes[ghostindex]<=0:
            if (distancetoGhost<2):
                score -= 500
            score = score + 1/distancetoFood +distancetoGhost/30

        else :
            if (distancetoGhost < 4):
                score += 1000
            if (distancetoGhost < 1):
                score += 10000
            score = score + 1/ distancetoFood
        return score
       # return successorGameState.getScore()

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
        "*** YOUR CODE HERE ***"
        """
        def max_value(state, depth, agentIndex):
            #if depth == self.depth or state.isWin() or state.isLose():
            #    return self.evaluationFunction(state)
            value = float("-inf")
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                nextstate = state.generateSuccessor(0, action)
                temp = minimax_value(nextstate, depth, agentIndex+1)
                if temp > value:
                    value = temp
                    move = action
                #value = max(value, minimax_value(nextstate, depth, agentIndex+1))
            return value

        def min_value(state, depth, agentIndex):
            #if depth == self.depth or state.isWin() or state.isLose():
              #  return self.evaluationFunction(state)
            value = float("inf")
            legalActions = state.getLegalActions(agentIndex)
            numofGhost = state.getNumAgents()-1
            if agentIndex == numofGhost:
                for action in legalActions:
                    nextstate = state.generateSuccessor(agentIndex, action)
                    temp = minimax_value(nextstate, depth+1, 0)
                    if temp<value:
                        value = temp
                        move = action
                    #value = min(value, minimax_value(nextstate, depth+1, 0))
            else :
                for action in legalActions:
                    nextstate = state.generateSuccessor(agentIndex, action)
                    temp = min_value(nextstate, depth, agentIndex+1)
                    if temp < value:
                        value = temp
                        move = action
                    # value = min(value, min_value(nextstate, depth, agentIndex+1))
            return value

        def minimax_value(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex != 0:
                return min_value(state, depth, 1)
            else:
                return max_value(state, depth, 0)

        legalMoves = gameState.getLegalActions()
        move = Directions.STOP

        value = float("-inf")

        for action in legalMoves:
            temp = minimax_value(gameState.generateSuccessor(0, action), 0, 1)
            if temp > value:
                value = temp
                move = action

        return move
        """
        """
        def maximizer(state, depth):
            move = Directions.STOP
            if depth == self.depth or state.isWin() or state.isLose():
                return [self.evaluationFunction(state), move]
            value = float("-inf")
            legalMoves = state.getLegalActions()
            for action in legalMoves:
                temp = minimizer(state.generateSuccessor(0,action), depth, 1)
                if temp[0]>value:
                    value = temp[0]
                    move = action
            return [value, move]

        def minimizer(state,depth,agentIndex):
            move = Directions.STOP
            if depth==self.depth or state.isWin() or state.isLose():
                return [self.evaluationFunction(state),move]
            value = float("inf")
            legalMoves = state.getLegalActions(agentIndex)
            if agentIndex == state.getNumAgents()-1:
                for action in legalMoves:
                    temp = maximizer(state.generateSuccessor(agentIndex, action), depth+1)
                    if temp[0]<value:
                        value = temp[0]
                        move =action
                   # value = min(value, maximizer(state.generateSuccessor(agentIndex, action), depth+1))
            else:
                for action in legalMoves:
                    temp = minimizer(state.generateSuccessor(agentIndex, action),depth, agentIndex+1)
                    if temp[0]<value:
                        value = temp[0]
                        move = action
                    #value = min(value, minimizer(state.generateSuccessor(agentIndex, action),depth, agentIndex+1))
            return [value, move]

        legalMoves = gameState.getLegalActions()
        move = Directions.STOP

        value = float("-inf")

        for action in legalMoves:
            temp = minimizer(gameState.generateSuccessor(0,action),0,1)
            if temp[0]>value:
                value = temp[0]
                move = action

        return move

        """



        def maximizer(state, depth):
            move = Directions.STOP
            #if depth == self.depth or state.isWin() or state.isLose():
                #return [self.evaluationFunction(state), move]
            value = float("-inf")
            legalMoves = state.getLegalActions()
            for action in legalMoves:
                temp = minimax_value(state.generateSuccessor(0,action), depth, 1)
                if temp[0]>value:
                    value = temp[0]
                    move = action
            return [value, move]

        def minimizer(state,depth,agentIndex):
            move = Directions.STOP
            if depth==self.depth or state.isWin() or state.isLose():
                return [self.evaluationFunction(state),move]
            value = float("inf")
            legalMoves = state.getLegalActions(agentIndex)
            if agentIndex == state.getNumAgents()-1:
                for action in legalMoves:
                    temp = minimax_value(state.generateSuccessor(agentIndex, action), depth+1, 0)
                    if temp[0]<value:
                        value = temp[0]
                        move =action
                   # value = min(value, maximizer(state.generateSuccessor(agentIndex, action), depth+1))
            else:
                for action in legalMoves:
                    temp = minimizer(state.generateSuccessor(agentIndex, action),depth, agentIndex+1)
                    if temp[0]<value:
                        value = temp[0]
                        move = action
                    #value = min(value, minimizer(state.generateSuccessor(agentIndex, action),depth, agentIndex+1))
            return [value, move]

        def minimax_value(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return [self.evaluationFunction(state),]
            if agentIndex != 0:
                return minimizer(state, depth, 1)
            else:
                return maximizer(state, depth)

        legalMoves = gameState.getLegalActions()
        move = Directions.STOP

        value = float("-inf")

        for action in legalMoves:
            temp = minimax_value(gameState.generateSuccessor(0,action),0,1)
            if temp[0]>value:
                value = temp[0]
                move = action

        return move


        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
