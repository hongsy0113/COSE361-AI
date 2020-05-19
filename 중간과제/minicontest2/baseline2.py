# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from distanceCalculator import Distancer


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MyAgent', second='MyAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class MyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)
        distancer = Distancer(gameState.data.layout)
        distancer.getMazeDistances()

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)
        scores = [self.evaluationFunction(gameState, action) for action in actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return actions[chosenIndex]

        '''
    You should change this in your own agent.
    '''

        # return random.choice(actions)

    def amIPacman(self, gameState):
        # agentstate = gameState.agentState
        if gameState.getAgentState(self.index).isPacman:
            return True
        else:
            return False

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def Vfunction(self, curS):
        result = 0

        maxvalue = float("-inf")
        actions = curS.getLegalActions(self.index)
        for action in actions:
            nextS = D
            result = max(maxvalue, 0.8 * Vfunction())

    # 팩맨의 reward 함수
    def pacmanRfunction(self, curS,nextS, agentIndex):
        # MDP R 함수
        result = 0
        # curS 는 S, nextS 는 S'
        eaten = 0
        #curS = self.getPreviousObservation()

        curPos = curS.getPosition(agentIndex)
        nextPos = nextS.getPosition(agentIndex)
        deposit = curS.getAgentState(agentIndex).numCarrying
        myCapsules = nextS.getAgentState.getCapsules(curS).asList()
        isRedTeam = curS.isOnRedTeam(agentIndex)
        # S'에서 상대의 상태 리스트
        opponents = [nextS.getAgentState(i) for i in self.getOpponents(nextS)]
        # S'에서 고스트인 상대의 상태 리스트
        ghost_opponents = [opponent for opponent in opponents if
                           (opponent.isPacman) != 1 and opponent.getPosition() != None]
        pos_scared_opponents = [opponent.getPosition() for opponent in ghost_opponents if opponent.scaredTimer > 0]
        # for p in pac_opponents:
        # if p.getposition() == curPos:
        #    eaten += 1
        # 갑자기 위치가 확 달라지면 먹힌거
        if self.getMazeDistance(curPos, nextPos) > 1:
            eaten += 1
        # 팩맨이 음식들을 무사히 가져왔다면 보상 크게
        if deposit > 0:
            if nextS.getAgentState(agentIndex).isPacman == 0 and eaten == 0:
                result += 20 * deposit
                return result
            if nextS.getAgentState(agentIndex).isPacman == 0 and eaten == 1:
                result -= 20 * deposit
                return result
        # 기본적으로 먹히지 않는 다면 R = 1
        if eaten == 0:
            result += 1
            if nextS.getAgentState(agentIndex).numCarrying > deposit:
                result += 5
            if nextPos in myCapsules:
                result += 7
            if nextPos in pos_scared_opponents:
                result += 10
        # 먹히면 R= -10
        else:
            result - + 10
            return result

        return result

    def ghostRfunction(self, curS, nextS, agentIndex):
        # MDP R 함수
        result = 0
        eaten = 0
        # curS 는 S, nextS 는 S'
        #curS = self.getPreviousObservation()
        curPos = curS.getAgentPosition(agentIndex)
        nextPos = nextS.getAgentPosition(agentIndex)
        # S' 지켜야할 음식들
        nextFoods = self.getFoodYouAreDefending(nextS).asList()
        # S' 남은 음식들 개수
        num_nextFoods = len(nextFoods)
        # S 남은 음식들 개수
        num_curFoods = len(self.getFoodYouAreDefending(curS).asList())
        # S에서 상대의 상태 리스트
        cur_opponents = [curS.getAgentState(i) for i in self.getOpponents(curS)]
        # S에서 팩맨인 상대의 상태 리스트
        cur_pac_opponents = [opponent for opponent in cur_opponents if
                             (opponent.isPacman) == 1 and opponent.getPosition() != None]
        # S'에서 상대의 상태 리스트
        next_opponents = [nextS.getAgentState(i) for i in self.getOpponents(nextS)]
        # S'에서 팩맨인 상대의 상태 리스트
        next_pac_opponents = [opponent for opponent in next_opponents if
                              (opponent.isPacman) == 1 and opponent.getPosition() != None]

        # S'에서 상대 팩맨 수가 줄었다면 상대가 도망을 갔거나 내가 먹은거
        if len(cur_pac_opponents) > len(next_pac_opponents):
            # 팩맨 수가 줄었는데 남은 음식수는 늘었다면 점수를 잃지 않는데 성공
            if num_curFoods < num_nextFoods:
                result += (num_nextFoods - num_curFoods + 1) * 10
            elif num_curFoods == num_nextFoods:
                # 남은 음식 차이가 없고 점수 차이가 없다면 상대가 음식 없이 도망을 갔거나 음식 없는 팩맨을 잡은 것
                if self.getScore(curS) == self.getScore(nextS):
                    result += 10
                else:
                    result += (self.getScore(nextS) - self.getScore(curS)) * 100
        # 상대가 음식을 먹었다면
        if num_curFoods > num_nextFoods:
            result += (num_nextFoods - numcurFoods)
        # 먹힌거
        if self.getMazeDistance(curPos, nextPos) > 1:
            result -= 50

        return result

    def evaluationFunction(self, currentgameState, action):
        score = 0
        nextS = self.getSuccessor(currentgameState, action)

        if self.amIPacman(currentgameState):
            score += self.pacmanRfunction(currentgameState,nextS, self.index)
        else:
            score += self.ghostRfunction(currentgameState, nextS, self.index)
        return score


class activeAgent(MyAgent):

    def evaluation(self, currentState, action):
        # 내가 팩맨이라면######################################

        if self.amIPacman(currentState):
            # 상대의 공격 여부 확인
            num_opp_attack = 0
            for index in getOpponents(currentState):
                if currentState.getAgentState(index).isPacman:
                    num_opp_attack += 1
            # 상대 수비수 한명이라면##################
            if num_opp_attack == 1:
                pass
