# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    start = problem.getStartState() # get start state
    stack = util.Stack()  # use stack because dfs is LIFO
    stack.push((start, [])) # stack consists of state and path to get the state
    explored = {start} # explored is a set of all states that the search function explored
    while True: # loop until stack is empty
        if stack.isEmpty() == True: break
        state, path = stack.pop()  # get data from stack (LIFO)
        explored.add(state) # add the state to explored set
        if problem.isGoalState(state):
            return path             # if the state is goal, return total actions to get the state
        for nextstate, action, cost in problem.getSuccessors(state):
            if not nextstate in explored: # check whether the state has been explored
                stack.push((nextstate,path+[action])) # push next state and total actions to get the state
    return[]
    # return empty list if there is no way to get goalstate


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    queue = util.Queue()    # use Queue because bfs is FIFO
    queue.push((start,[]))
    explored = {start}
    while True:
        if queue.isEmpty() == True: break
        state, path = queue.pop()
        explored.add(state)
        if problem.isGoalState(state):
            return path
        for nextstate, action, cost in problem.getSuccessors(state):
            if not nextstate in explored:
                queue.push((nextstate, path + [action]))
                explored.add(nextstate)
    return []
    # similar with dfs

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    Pqueue = util.PriorityQueue() # use PriorityQueue because ucs needs comparison
    Pqueue.push((start, []),0) # PriorityQueue consists of state, path to get the state and priority
    explored = {start:0} # explored is dictionary value signify explored states and the total min cost to get the state
    while True:
        if Pqueue.isEmpty() == True: break
        state, path= Pqueue.pop()
        explored[state] = problem.getCostOfActions(path) # get cost of the state by using this function
        if problem.isGoalState(state):
            return path
        for nextstate, action, cost in problem.getSuccessors(state):
            if (not nextstate in explored) or ((nextstate in explored ) and (explored[state]+cost<explored[nextstate])):
                # check whether next state has been explored and if new cost is less than stored total cost
                explored[nextstate] = explored[state] + cost  # store the cost to get the next state
                Pqueue.push((nextstate, path + [action]), explored[nextstate])
                # push new data into PriorityQueue including next state, actions, and cost as a priority
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    Pqueue = util.PriorityQueue() # use priorityqueue
    Pqueue.push((start, []), 0)
    explored = set()
    while True:
        if Pqueue.isEmpty() == True: break
        state, path = Pqueue.pop()
        if state in explored: continue
        explored.add(state)
        if problem.isGoalState(state):
            return path
        for nextstate, action, cost in problem.getSuccessors(state):
            if not nextstate in explored:
                priority = cost+ problem.getCostOfActions(path)+ (heuristic(nextstate, problem=problem))
                Pqueue.push((nextstate, path + [action]), priority)
    return []
    # similar with ucs but cost should include heuristic function

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
