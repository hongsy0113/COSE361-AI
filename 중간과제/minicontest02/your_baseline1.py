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
               first = 'passiveAgent', second = 'activeAgent'):

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):


  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)
    distancer = Distancer(gameState.data.layout)
    distancer.getMazeDistances()
    self.start = gameState.getAgentPosition(self.index)

  # 다음 상태를 빠르게 구하기 위한 함수
  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    return successor

  # 내가 팩맨인지 아닌지를 빠르게 구하기 위한 함수
  def amIPacman (self, gameState):
    #agentstate = gameState.agentState
    if gameState.getAgentState(self.index).isPacman:
      return True
    else:
      return False

class activeAgent(DummyAgent):
  # 평가함수를 통해 active 하게 공격을 나가는 reflex agent
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    # 가능한 움직임 리스트로 반환
    actions = gameState.getLegalActions(self.index)

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in actions]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]

  def evaluationFunction(self, currentgameState, action):
    #다음상태
    nextstate = self.getSuccessor(currentgameState, action)
    # 현재위치와 다음위치
    nowPos = currentgameState.getAgentPosition(self.index)
    newPos = nextstate.getAgentPosition(self.index)
    # 점수, feature, 가중치 초기화
    score = 0
    features = util.Counter()
    weights = self.getWeights()

    foodList = self.getFood(nextstate).asList()
    num_Food = len(foodList)
    # 남은 음식의 개수 저장
    features['num_Food'] = num_Food

    minFoodDist = float("inf")
    minGhostDist = float("inf")
    deposit = currentgameState.getAgentState(self.index).numCarrying
    # 현재 먹어서 가지고 있는 음식의 개수 저장
    features['carrying'] = deposit

    # 음식 먹었는지 여부 저장
    if self.getFood(currentgameState)[newPos[0]][newPos[1]] == True:
      features['eat'] = 1

    for food in foodList:
      minFoodDist = min(self.getMazeDistance(newPos, food), minFoodDist)
    # 음식의 최소거리 저장
    features['DisttoFood'] = 1/minFoodDist

    # 상대의 상태 리스트
    opponents = [nextstate.getAgentState(i) for i in self.getOpponents(nextstate)]
    # 고스트인 상대의 상태 리스트
    ghost_opponents = [opponent for opponent in opponents if
                     (opponent.isPacman) != 1 and opponent.getPosition() != None]

    for p in ghost_opponents:
      temp = self.getMazeDistance(newPos, p.getPosition())
      if temp < minGhostDist:
        minGhostDist = temp
        closestGhost = p
    # 고스트 최소 거리 저장
    features['DisttoGhost'] = minGhostDist

    # 점수를 무사히 가져왔는지에 대한 정보 저장
    getpoint = self.getScore(nextstate)-self.getScore(currentgameState)
    features['getpoint'] =getpoint

    # 캡슐 먹었는지 여부 저장
    if len(self.getCapsules(currentgameState))-len(self.getCapsules(nextstate)):
      features['eatCapsules'] = 1

    if len(ghost_opponents) >0 and closestGhost.scaredTimer == 0:
      # 고스트와의 거리가 가까울수록 마이너스 점수를 크게 부여한다
      if minGhostDist <5:
        score = minGhostDist
        if minGhostDist<=2:
          score -= 1000
      score += features * weights

    # 고스트가 scared 상태라면 영향 받지 않는다
    else:
      score += minFoodDist * (-10) + 100 * features['eat']

    # 두 개 이상 음식을 먹었다면 무리하지 않고 점수를 위해 복귀하도록 점수 부여
    if currentgameState.getAgentState(self.index).numCarrying >=2:
      myFoods = self.getFoodYouAreDefending(currentgameState).asList()
      myFoodDist = self.getMazeDistance(myFoods[0], newPos)
      # 우리팀 진영으로 다시 오게 하기 위해 우리팀의 음식 정보 사용
      score += (-500)*myFoodDist + minGhostDist
      return score
    return score

  # 가중치 반환하는 함수
  def getWeights(self):
    return {'eat': 100, 'DisttoFood': 100, 'DisttoGhost': 1/25, 'num_Food' : -5, 'carrying': 10, 'eatCapsules': 100, 'getpoint' : 500}

class passiveAgent(DummyAgent):
  # 평가함수를 통해 수비를 적극적으로 하는 reflex agent
  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    # 가능한 움직임 리스트로 반환
    actions = gameState.getLegalActions(self.index)

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in actions]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return actions[chosenIndex]

  def evaluationFunction(self, currentgameState, action):
    #다음상태
    nextstate = self.getSuccessor(currentgameState, action)
    # 현재위치와 다음위치
    nowPos = currentgameState.getAgentPosition(self.index)
    newPos = nextstate.getAgentPosition(self.index)

    score = 0
    features = util.Counter()
    weights = self.getWeights()

    notAttack = 1   # 현재 내가 수비중인지에 대한 변수
    isstop = 0      # 멈춰있는 상태를 지양하기 위한 변수
    isrev = 0       # 역방향으로 가는 행동을 지양하기 위한 변수
    if action == Directions.STOP:
      isstop = 1
    if action == Directions.REVERSE[currentgameState.getAgentState(self.index).configuration.direction]:
      isrev = 1
    if self.amIPacman(currentgameState):
      notAttack = 0
    # 현재 내가 수비중인지에 대한 정보 저장
    features ['notAttack'] = notAttack

    foodList = self.getFoodYouAreDefending(nextstate).asList()
    num_Food = len(foodList)

    minFoodDist = float("inf")
    minPacDist = float("inf")

    # 음식의 최소거리 저장
    features['foodsleft'] = num_Food
    # 상대의 상태 리스트
    opponents = [nextstate.getAgentState(i) for i in self.getOpponents(nextstate)]
    # 팩맨인 상대의 상태 리스트
    pac_opponents = [opponent for opponent in opponents if
                     (opponent.isPacman) == 1 and opponent.getPosition() != None]
    num_invaders = len (pac_opponents)
    # 공격중인 상대의 개수 정보 저장
    features['invaders'] = num_invaders
    for p in pac_opponents:
      temp = self.getMazeDistance(newPos, p.getPosition())
      if temp < minPacDist:
        minPacDist = temp
        closestGhost = p

    # 상대 팩만과의 최소 거리 저장
    features['DisttoOpponent'] = minPacDist

    # 공격중인 상대가 없다면 큰 의미없이 랜덤하게 움직임
    if num_invaders == 0:
      score += (-10) * isstop + (-10) * isrev + 100
    # 공격중인 상대가 한명이라도 있다면
    else :
      if currentgameState.getAgentState(self.index).scaredTimer == 0:
        score += features * weights
      # 내가 scared 상태라면 먹히지 않도록 조심한다. 하지만 무조건 멀리 도망갈 필요는 없도록 구현
      else:
        if (minPacDist <=3):
          score -= 10000
          score += minFoodDist * (-10) + 100 * features['eat']
        else:
          score += features * weights

    return score

  def getWeights(self):
    return {'notAttack': 10, 'foodsleft': 5, 'DisttoOpponent': -30,'isstop': -10, 'isrev':-10, 'invaders': -10}