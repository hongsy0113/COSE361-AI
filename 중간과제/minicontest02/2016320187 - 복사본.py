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
               first='BestAgent', second='BestAgent'):
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

class BestAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
      CaptureAgent.registerInitialState(self, gameState)
      distancer = Distancer(gameState.data.layout)
      distancer.getMazeDistances()
      self.start = gameState.getAgentPosition(self.index)

    def chooseAction(self, gameState):
        # 가능한 움직임 리스트로 반환
      actions = gameState.getLegalActions(self.index)

      # Choose one of the best actions
      # 상대의 상태 리스트
      opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      # 고스트인 상대의 상태 리스트
      pac_opponents = [opponent for opponent in opponents if
                       (opponent.isPacman) == 1 and opponent.getPosition() != None]
      num_invaders = len(pac_opponents)

      # 둘중에 누가 수비할 지 정하자
      teamIndex = self.getTeam(gameState)
      myIndex = self.index
      teamIndex.remove(myIndex)
      yourIndex = teamIndex[0]
      myPos = gameState.getAgentPosition(myIndex)
      yourPos = gameState.getAgentPosition(yourIndex)



      if num_invaders == 0:
          #scores = [self.attack_evaluate(gameState, action) for action in actions]
          move = activeAgent.chooseAction(self, gameState)
          return move
      elif num_invaders ==1:
          invaderPos = pac_opponents[0].getPosition()
          myDist = self.getMazeDistance(myPos, invaderPos)
          yourDist = self.getMazeDistance(yourPos, invaderPos)

          if myDist<= yourDist :
            move = passiveAgent.chooseAction(self,gameState)
          else:
            move = activeAgent.chooseAction(self,gameState)
          return move
      elif num_invaders ==2:
          move = passiveAgent.chooseAction(self, gameState)
          return move
      #scores = [self.evaluationFunction(gameState, action) for action in actions]
      #bestScore = max(scores)
      #bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
      #chosenIndex = random.choice(bestIndices) # Pick randomly among the best
      #return actions[chosenIndex]


    def amIPacman(self, gameState):
        # agentstate = gameState.agentState
        if gameState.getAgentState(self.index).isPacman:
            return True
        else:
            return False

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def teamState(self, gameState):
        # 우리팀의 상태 리스트
        teams = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        return teams

    def pacman_teamState(self, gameState):
        teams = self.teamState(gameState)
        pac_teams = [t for t in teams if (t.isPacman) == 1 and t.getPosition() != None]
        return pac_teams

    def ghost_teamState(self, gameState):
        teams = self.teamState(gameState)
        gho_teams = [t for t in teams if (t.isPacman) != 1 and t.getPosition() != None]
        return gho_teams

    def opponentState(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        return opponents

    def pacman_opponentState(self, gameState):
        opponents = self.opponentState(gameState)
        pac_opps = [t for t in opponents if (t.isPacman) == 1 and t.getPosition() != None]
        return pac_opps

    def ghost_opponentState(self, gameState):
        opponents = self.opponentState(gameState)
        gho_opps = [t for t in opponents if (t.isPacman) != 1 and t.getPosition() != None]
        return gho_opps


class activeAgent(BestAgent):

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """

        actions = gameState.getLegalActions(self.index)
        '''
        You should change this in your own agent.
        '''
        if gameState.isOnRedTeam(self.index):
            isRed = 1
            standard = 13
        else:
            isRed = 0
            standard = 18
        myPos = gameState.getAgentPosition(self.index)

        # 내가 고스트라면
        if self.amIPacman(gameState)==1:
        #if (isRed == 1 and myPos[0] >= standard) or (isRed == 0 and myPos[0] <= standard):
            result = minmaxAgent.getAction(self, gameState)
            return result[1]


        else:
            scores = [activeAgent.ghost_evaluation(self,gameState, action) for action in actions]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
            return actions[chosenIndex]

    def ghost_evaluation(self, currentState, action):
        # 내가 고스트라면######################################
        nextstate = self.getSuccessor(currentState, action)
        num_team_pac = len(self.pacman_teamState(currentState))
        num_team_gho = len(self.ghost_teamState(currentState))
        num_opp_pac = len(self.pacman_opponentState(currentState))
        num_opp_gho = len(self.ghost_opponentState(currentState))

        newPos = nextstate.getAgentPosition(self.index)
        score = 0

        # 공격중인 상대가 하나 이하라면 나는 그냥 공격
        if num_opp_pac <= 1:
            minFoodDist = float("inf")
            opp_minFoodDist = float("inf")
            total_FoodDist = 0
            isstop = 0
            isrev = 0
            foodList = self.getFood(nextstate).asList()
            for food in foodList:
                temp = self.getMazeDistance(newPos, food)
                total_FoodDist += temp
                if temp < minFoodDist:
                    minFoodDist = temp
                    minFood = food
            if(len(foodList)) == 1:
              avg_FoodDist = (total_FoodDist - minFoodDist)
            else:
              avg_FoodDist= (total_FoodDist - minFoodDist) / len(foodList) - 1
            # foodList.remove(minFood)

            for ghost in self.ghost_opponentState(nextstate):
                ghostPos = ghost.getPosition()
                temp2 = self.getMazeDistance(ghostPos, minFood)
                if temp2 < opp_minFoodDist:
                    opp_minFoodDist = temp2
                    closeGhost = ghostPos
                # minFoodDist = min(self.getMazeDistance(newPos, food), minFoodDist)
            if action == Directions.STOP:
                isstop = 1
            if action == Directions.REVERSE[currentState.getAgentState(self.index).configuration.direction]:
                isrev = 1
            # 내가 먼저 먹으려는 음식 근처에 상대가 있다면 조심
            clostghostDist = self.getMazeDistance(closeGhost, newPos)
            if (opp_minFoodDist - minFoodDist < 0 or opp_minFoodDist < 3 or clostghostDist < 6) and minFoodDist < 7:
                foodList.remove(minFood)
                secondFoodDistance = float("inf")
                for food in foodList:
                    temp = self.getMazeDistance(newPos, food)
                    if temp < secondFoodDistance:
                        secondFoodDistance = temp
                        secondminFood = food
                if currentState.isOnRedTeam(self.index):
                    score += (-1 / 2) * abs(newPos[0] - 16)  # +(-1) * isstop3*clostghostDist
                    # score += -1 * isstop + -1 * isrev + -2 * avg_FoodDist
                else:
                    score += (-1 / 2) * abs(newPos[0] - 15)  # + (-1) * abs(newPos[0] - 15)# +(-1) * isstop
                # score += 1/ secondFoodDistance + self.getMazeDistance(closeGhost, newPos)/20
            else:
                score += -1 * isstop + -1 * isrev + (-1 / 2) * abs(newPos[0] - 16) + -2 * avg_FoodDist
        # 상대 공격 두명이면 같이 수비
        else:
            opponents = [nextstate.getAgentState(i) for i in self.getOpponents(nextstate)]
            pac_opponents = [opponent for opponent in opponents if opponent.isPacman and opponent.getPosition() != None]
            num_pac_opponents = len(pac_opponents)
            minOppDist = float("inf")

            for p in pac_opponents:
                if newPos == p.getPosition():
                    score += 10000
                minOppDist = min(self.getMazeDistance(newPos, p.getPosition()), minOppDist)
            if (nextstate.getAgentState(self.index).scaredTimer != 0):
                if (minOppDist <= 2): score -= 1000000
                score += num_pac_opponents * -100
            else:
                score += minOppDist * -100
        return score

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features


class minmaxAgent(BestAgent):

    def evaluationFunction(self, currentgameState):
        scores = []
        features = util.Counter()
        weights = minmaxAgent.getmMWeights(self)
        actions = currentgameState.getLegalActions(self.index)
        """
        for action in actions:
            # 다음상태
            nextstate = self.getSuccessor(currentgameState, action)
            # 현재위치와 다음위치
            nowPos = currentgameState.getAgentPosition(self.index)
            newPos = nextstate.getAgentPosition(self.index)
            # 점수, feature, 가중치 초기화

            score = self.getScore(currentgameState) * 500
            features = util.Counter()
            weights = minmaxAgent.attack_getWeights(self)

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
            features['DisttoFood'] = 1 / minFoodDist

            # 상대의 상태 리스트
            opponents = [nextstate.getAgentState(i) for i in self.getOpponents(nextstate)]
            # 고스트인 상대의 상태 리스트
            pac_opponents = [opponent for opponent in opponents if
                             (opponent.isPacman) != 1 and opponent.getPosition() != None]
            for p in pac_opponents:
                temp = self.getMazeDistance(newPos, p.getPosition())
                if temp < minGhostDist:
                    minGhostDist = temp
                    closestGhost = p
            # 고스트 최소 거리 저장
            features['DisttoGhost'] = minGhostDist

            # 점수를 무사히 가져왔는지에 대한 정보 저장
            getpoint = self.getScore(nextstate) - self.getScore(currentgameState)
            features['getpoint'] = getpoint

            # 캡슐 먹었는지 여부 저장
            if len(self.getCapsules(currentgameState)) - len(self.getCapsules(nextstate)):
                features['eatCapsules'] = 1
            score += features * weights
            if closestGhost.scaredTimer == 0:
                # 고스트와의 거리가 가까울수록 마이너스 점수를 크게 부여한다
                if minGhostDist < 5:
                    myFoods = self.getFoodYouAreDefending(currentgameState).asList()
                    myFoodDist = self.getMazeDistance(myFoods[0], newPos)

                    #score = minGhostDist * 10 + (-1)*myFoodDist
                    score -= 5000

                    if minGhostDist <= 2:
                        score -= 10000
                #score += features * weights

            # 고스트가 scared 상태라면 영향 받지 않는다
            else:
                score += minFoodDist * (-10) + 100 * features['eat'] + 100 * features['carrying']
            scores.append(score)
        bestscore = max(scores)
        """
        point = self.getScore(currentgameState)
        myPos = currentgameState.getAgentPosition(self.index)
        ghostopp = self.ghost_opponentState(currentgameState)
        minGhostDist = 15
        minGhost = self.opponentState(currentgameState)[0]

        foodList = self.getFood(currentgameState).asList()
        num_Food = len(foodList)
        minFoodDist = float("inf")

        for food in foodList:
            minFoodDist = min(self.getMazeDistance(myPos, food), minFoodDist)
        for e in ghostopp:
            temp = self.getMazeDistance(myPos, e.getPosition())
            if temp < minGhostDist:
                minGhostDist = temp
                minGhost = e
        features['point'] = point
        features['minGhostDist'] = minGhostDist
        features['minFoodDist'] =1/ minFoodDist
        features['deposit'] = currentgameState.getAgentState(self.index).numCarrying

        score = features * weights

        myFoods = self.getFoodYouAreDefending(currentgameState).asList()
        myFoodDist = self.getMazeDistance(myFoods[0], myPos)

        """
        
          if minGhostDist <= 5:
            score = 2 * minGhostDist + 1 / minFoodDist
            # score -= 1000
          if minGhostDist <= 3:
            
            # 우리팀 진영으로 다시 오게 하기 위해 우리팀의 음식 정보 사용
            score = (-1) * myFoodDist + 2*minGhostDist
            #score = 2 * minGhostDist
            # score -= 10000
            """
        if minGhost.scaredTimer == 0:
          if minGhostDist <= 4:
            score = 20 * minGhostDist + 1 / myFoodDist + 1/ minFoodDist
            # score -= 1000
        else :

          score = features['minFoodDist'] * 1/25 + features ['deposit'] * 10 + features['getpoint'] *30
          score += 10000
        return [score, ]

    def attack_getWeights(self):
        return {'eat': 10, 'DisttoFood': 10, 'DisttoGhost': 1 / 30, 'num_Food': -1, 'carrying': 10,
                'eatCapsules': 10, 'getpoint': 100}
    def getmMWeights(self):
        return {'point': 100, 'minGhostDist': 1/25, 'minFoodDist': 50, 'deposit': 50, 'getpoint':30}

    def getAction(self, gameState):

        numofGhost = len(self.ghost_opponentState(gameState))
        if numofGhost == 2:
            d = 2
        elif numofGhost == 1:
            d = 3

        # opponents = self.opponentState(gameState)
        # gho_opps =  [t for t in opponents if(t.isPacman)!=1 and t.getPosition()!= None]

        # opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman != 1]

        def max_value(state, depth, agentIndex, alpha, beta):
            if depth == d:
                return minmaxAgent.evaluationFunction(self, state)
            # initialize v = - inf
            value = float("-inf")
            # move를 stop으로 초기화
            move = Directions.STOP
            # 현재 max agent가 (팩맨)이 갈 수 있는 모든 움직임 저장
            legalActions = state.getLegalActions(agentIndex)
            # for each successor of current state
            for action in legalActions:
                nextstate = state.generateSuccessor(agentIndex, action)
                # v = max(v, value(successor)
                temp = min_value(nextstate, depth + 1, self.getOpponents(state)[0], alpha, beta)
                if temp[0] > value:
                    value = temp[0]
                    move = action
                if value > beta:
                    return [value, move]
                alpha = max(alpha, value)
            # 최대 value 와 그때의 action return
            return [value, move]

        def min_value(state, depth, agentIndex, alpha, beta):

            if depth == d:
                return minmaxAgent.evaluationFunction(self, state)
            numofGhost = len(self.ghost_opponentState(state))
            move = Directions.STOP
            value = float("inf")
            legalActions = state.getLegalActions(agentIndex)
            if numofGhost > 1:
                if agentIndex == self.getOpponents(state)[numofGhost - 1]:
                    for action in legalActions:
                        nextstate = state.generateSuccessor(agentIndex, action)
                        # v = min(v, value(successor)
                        temp = max_value(nextstate, depth + 1, self.index, alpha, beta)
                        if temp[0] < value:
                            value = temp[0]
                            move = action
                        if value < alpha:
                            return [value, move]
                        beta = min(beta, value)
                # 마지막 고스트가 아닐때
                else:
                    for action in legalActions:
                        nextstate = state.generateSuccessor(agentIndex, action)
                        # 점수들의 최소값 중에서의 최소값 구하는 과정
                        # 다음 유령으로 넘어감
                        temp = min_value(nextstate, depth, self.getOpponents(state)[numofGhost - 1], alpha, beta)
                        if temp[0] < value:
                            value = temp[0]
                            move = action
                        if value < alpha:
                            return [value, move]
                        beta = min(beta, value)
            else:
                for action in legalActions:
                    nextstate = state.generateSuccessor(agentIndex, action)
                    # v = min(v, value(successor)
                    temp = max_value(nextstate, depth + 1, self.index, alpha, beta)
                    if temp[0] < value:
                        value = temp[0]
                        move = action
                    if value < alpha:
                        return [value, move]
                    beta = min(beta, value)
            # 최소 value 와 그때의 action return
            return [value, move]

        legalMoves = gameState.getLegalActions(self.index)
        value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in legalMoves:
            nextstate = self.getSuccessor(gameState, action)
            temp = min_value(nextstate, 0, self.getOpponents(gameState)[0], alpha, beta)
            if temp[0] > value:
                value = temp[0]
                move = action
            alpha = max(alpha, value)
        return [value, move]


class passiveAgent(BestAgent):

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """

        # test = self.getMazeDistance((1,1), (3,1))
        # print(test)
        # self.debugDraw([(1,1),(3,1)], [0,1,1], True)
        actions = gameState.getLegalActions(self.index)
        scores = [passiveAgent.evaluationFunction(self, gameState, action) for action in actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return actions[chosenIndex]

    def evaluationFunction(self, currentgameState, action):

        nextstate = self.getSuccessor(currentgameState, action)

        nowPos = currentgameState.getAgentPosition(self.index)
        newPos = nextstate.getAgentPosition(self.index)
        score = 0

        pac_opponents = self.pacman_opponentState(currentgameState)
        num_pac_opponents = len(pac_opponents)
        minOppDist = float("inf")

        if len(pac_opponents) != 0:
            for p in pac_opponents:
                if newPos == p.getPosition():
                    score += 10000
                minOppDist = min(self.getMazeDistance(newPos, p.getPosition()), minOppDist)
            if (nextstate.getAgentState(self.index).scaredTimer != 0):
                if (minOppDist <= 2):
                    score -= 1000000
                    score += minOppDist * -100
            else:
                score += minOppDist * -100
        else:
            isstop = 0
            isrev = 0
            if action == Directions.STOP:
                isstop = 1
            if action == Directions.REVERSE[currentgameState.getAgentState(self.index).configuration.direction]:
                isrev = 1
            score += -1 * isstop + -1 * isrev
        return score


"""
  if currentgameState.getAgentState(self.index).scaredTimer == 0:
    score += features * weights
  # 내가 scared 상태라면 먹히지 않도록 조심한다. 하지만 무조건 멀리 도망갈 필요는 없도록 구현
  else:
    if (minPacDist <= 3):
      score -= 10000
      score += minFoodDist * (-10) + 100 * features['eat']
    else:
      score += features * weights
"""