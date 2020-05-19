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
               first='activeAgent', second='passiveAgent'):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class MyAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)
        distancer = Distancer(gameState.data.layout)
        distancer.getMazeDistances()

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

    # 내가 팩맨인지 아닌지 빠르게 구하기 위한 함수
    def amIPacman(self, gameState):
        if gameState.getAgentState(self.index).isPacman:
            return True
        else:
            return False

    # 다음 상태를 빠르게 얻기위한 함수
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    # 우리팀의 상태 리스트 반환
    def teamState(self, gameState):
        teams = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        return teams

    # 팩맨인 우리팀의 상태 리스트 반환
    def pacman_teamState(self, gameState):
        teams = self.teamState(gameState)
        pac_teams = [t for t in teams if (t.isPacman) == 1 and t.getPosition() != None]
        return pac_teams

    # 고스트인 우리팀의 상태 리스트 반환
    def ghost_teamState(self, gameState):
        teams = self.teamState(gameState)
        gho_teams = [t for t in teams if (t.isPacman) != 1 and t.getPosition() != None]
        return gho_teams

    # 상대팀의 상태 리스트 반환
    def opponentState(self, gameState):
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        return opponents

    # 팩맨인 상대팀의 상태 리스트 반환
    def pacman_opponentState(self, gameState):
        opponents = self.opponentState(gameState)
        pac_opps = [t for t in opponents if (t.isPacman) == 1 and t.getPosition() != None]
        return pac_opps

    # 고스트인 상대팀의 상태 리스트 반환
    def ghost_opponentState(self, gameState):
        opponents = self.opponentState(gameState)
        gho_opps = [t for t in opponents if (t.isPacman) != 1 and t.getPosition() != None]
        return gho_opps

# minimax 알고리즘을 이용하여 공격을 적극적으로 하는 agent
class activeAgent(MyAgent):
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        # 나의 위치 정보
        myPos = gameState.getAgentPosition(self.index)

        # 내가 팩맨이라면 minimax 알고리즘 사용
        if self.amIPacman(gameState)==1:
            result = minmaxAgent.getAction(self, gameState)
            return result[1]

        # 고스트라면 평가함수를 통해 reflex 한 움직임 반환
        else:
            scores = [self.ghost_evaluation(gameState, action) for action in actions]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)
            return actions[chosenIndex]

    # 고스트일 때 행동을 결정해주는 평가 함수
    def ghost_evaluation(self, currentState, action):
        nextstate = self.getSuccessor(currentState, action)         # 다음 상태 정보
        newPos = nextstate.getAgentPosition(self.index)             # 다음 위치 정보
        num_team_pac = len(self.pacman_teamState(currentState))     # 우리팀 공격 수
        num_team_gho = len(self.ghost_teamState(currentState))      # 우리팀 수비 수
        num_opp_pac = len(self.pacman_opponentState(currentState))  # 상대팀 공격 수
        num_opp_gho = len(self.ghost_opponentState(currentState))   # 상대팀 수비 수

        score = 0

        # 공격중인 상대가 하나 이하라면 나는 공격
        if num_opp_pac <= 1:
            minFoodDist = float("inf")
            opp_minFoodDist = float("inf")
            minGhostDist = float("inf")
            total_FoodDist = 0
            isstop = 0      # 멈춰있는 상태를 지양하기 위한 변수
            isrev = 0       # 역방향으로 가는 행동을 지양하기 위한 변수
            if action == Directions.STOP:
                isstop = 1
            if action == Directions.REVERSE[currentState.getAgentState(self.index).configuration.direction]:
                isrev = 1
            foodList = self.getFood(nextstate).asList()
            # 가장 가까운 음식의 최소거리 구하기
            for food in foodList:
                temp = self.getMazeDistance(newPos, food)
                total_FoodDist += temp
                if temp < minFoodDist:
                    minFoodDist = temp
                    minFood = food
            
            # 가장 가까운 음식을 제외한 음식들과의 평균거리
            avg_FoodDist = (total_FoodDist - minFoodDist) / len(foodList) - 1

            # 가장 가까운 고스트와 그 거리 구하기
            # 가장 가까운 음식과 가장 가까운 고스트와 그 거리 구하기
            for ghost in self.ghost_opponentState(nextstate):
                ghostPos = ghost.getPosition()
                minGhostDist = min(self.getMazeDistance(newPos, ghost.getPosition()), minGhostDist)
                temp2 = self.getMazeDistance(ghostPos, minFood)
                if temp2 < opp_minFoodDist:
                    opp_minFoodDist = temp2
                    closeGhost = ghostPos

            # 내가 먼저 먹으려는 음식 근처에 상대가 있다면 조심
            if ((opp_minFoodDist - minFoodDist < 0 or opp_minFoodDist < 3 ) and minFoodDist < 7) or minGhostDist < 4:
                foodList.remove(minFood)
                secondFoodDistance = float("inf")
                for food in foodList:
                    temp = self.getMazeDistance(newPos, food)
                    if temp < secondFoodDistance:
                        secondFoodDistance = temp
                        secondminFood = food
                # 단순히 음식만 쫓아가면 먹힐 수도 있으므로 고스트와의 거리 등 여러 요소 고려
                score+= 1/secondFoodDistance + minGhostDist/2 + 1/minFoodDist + 1/avg_FoodDist
            else:
                # 음식들의 평균 거리를 줄이는 점수 반환
                score += -1 * isstop + -1 * isrev + -3 * avg_FoodDist + minGhostDist/30
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
            # 내가 scared 상태라면 먹히지 않도록 조심한다. 하지만 무조건 멀리 도망갈 필요는 없도록 구현
            if (nextstate.getAgentState(self.index).scaredTimer != 0):
                if (minOppDist <= 2): score -= 1000000
                score += num_pac_opponents * -100
            else:
                score += minOppDist * -100
        return score

# 하위 클래스로 minmaxagent
class minmaxAgent(MyAgent):

    # leaf node에서 사용하기 위한 평가함수
    def evaluationFunction(self, currentgameState):
        scores = []
        features = util.Counter()
        weights = minmaxAgent.getmMWeights(self)
        actions = currentgameState.getLegalActions(self.index)

        point = self.getScore(currentgameState) # 현재 점수
        myPos = currentgameState.getAgentPosition(self.index)   # 나의 위치
        ghostopp = self.ghost_opponentState(currentgameState)   # 고스트인 상대 정보
        minGhostDist = 50
        minGhost = self.opponentState(currentgameState)[0]      # 초기화

        foodList = self.getFood(currentgameState).asList()
        num_Food = len(foodList)                                # 음식들의 정보
        minFoodDist = float("inf")

        # 가장 가까운 음식과의 거리
        for food in foodList:
            minFoodDist = min(self.getMazeDistance(myPos, food), minFoodDist)
        # 가장 가까운 고스트와 그 거리
        for e in ghostopp:
            temp = self.getMazeDistance(myPos, e.getPosition())
            if temp < minGhostDist:
                minGhostDist = temp
                minGhost = e
        # 가장 가까운 고스트와의 맨해튼 거리
        ghostDist = abs(minGhost.getPosition()[0] - myPos[0]) + abs(minGhost.getPosition()[1] - myPos[1])
        # 평가 함수를 위한 features 저장
        features['point'] = point
        features['minGhostDist'] = minGhostDist
        features['minFoodDist'] =1/ minFoodDist
        features['deposit'] = currentgameState.getAgentState(self.index).numCarrying

        score = features * weights

        # 우리팀 진영으로 다시 오게 하기 위해 우리팀의 음식 정보 사용
        myFoods = self.getFoodYouAreDefending(currentgameState).asList()
        if len(myFoods) > 0:
            myFoodDist = self.getMazeDistance(myFoods[0],myPos)

        # 고스트의 상태가 scared 가 아니라면
        if minGhost.scaredTimer == 0:
            # 고스트와 거리가 가까울 땐 우선적으로 고스트와 거리를 벌리도록 가중치 변경
            if minGhostDist <= 4:
                score = - 1000*(1/minGhostDist) + 1/(minFoodDist*2)
                if minGhostDist <= 2:
                    score -= 1000
            # 먹히면 그 순간 고스트와의 맨해튼 거리가 멀어지므로 그때는 마이너스 크게
            if ghostDist >= currentgameState.data.layout.width/2:
                score -= 1000000
        # 고스트의 상태가 scared 라면 고스트에 영향 받지 않도록
        else:
            score = features['minFoodDist'] * 1  + features['deposit'] * 10 + features['point'] * 30
            score += 1000
        # 안정적인 점수 획득을 위해 음식을 네개 이상 먹었을 경우 복귀하도록 한다.
        if currentgameState.getAgentState(self.index).numCarrying >= 4:
            if len(myFoods) > 0:
                # 우리 팀의 음식 정보 이용
                score += (1/myFoodDist) *1000
            else :
                score = 200*features['deposit'] + 500*features['point'] + minGhostDist/10
        return [score, ]

    def getmMWeights(self):
        return {'point': 500, 'minGhostDist': 1 / 15, 'minFoodDist': 10, 'deposit': 20}

    def getAction(self, gameState):

        numofGhost = len(self.ghost_opponentState(gameState))
        # 계산 시간을 위해 상대 고스트에 따라 알고리즘의 깊이 조절
        if numofGhost != 1:
            d = 2
        elif numofGhost == 1:
            d = 3

        # maxagent(팩맨)의 함수
        # 알파, 베타 가지치기 이용
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

        # minagent(고스트)의 함수
        # 알파, 베타 가지치기 이용
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

# baseline1, 2 와 동일하게 reflex하게 수비를 하는 agent
class passiveAgent(MyAgent):

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        scores = [self.evaluationFunction(gameState, action) for action in actions]
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
            # 내가 scared 상태라면 먹히지 않도록 조심한다. 하지만 무조건 멀리 도망갈 필요는 없도록 구현
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