# Some algorithms I implemented in my AI class
# Not runnable, but work with required files.
# Files not included to protect integrity
class Node:
    """
    Represents a Node used in UCS
    """
    def __init__(self, previousNode, successor, action, stepCost):
        self.state = successor
        self.action = action
        self.stepCost = stepCost
        if (previousNode != None):
            self.totalCost = stepCost + previousNode.getTotalCost()
        else:
            self.totalCost = 0;
        self.previousNode = previousNode
       
    def getState(self):
        return self.state
    
    def getAction(self):
        return self.action
    
    def getStepCost(self):
        return self.stepCost
    
    def getTotalCost(self):
        return self.totalCost
    
    def getPreviousNode(self):
        return self.previousNode

def depthFirstSearch(problem):
    """
    Do a depth first search
    """
    visited = []
    actions = dict()
    pathMap = dict()
    stack = Stack()
    stack.push(problem.getStartState())
    found = False
    currentState = None
    while (not stack.isEmpty() and not found):
        currentState = stack.pop()
        if (problem.isGoalState(currentState)):
            found = True
        else:
            if (currentState not in visited):
                visited.append(currentState)
                nextStates = problem.getSuccessors(currentState)
                for state in nextStates:
                    if (state[0] not in visited):
                        stack.push(state[0])
                        actions[state[0]] = state[1]
                        pathMap[state[0]] = currentState
    
    finalPath = []
    """form the final path if one is found"""
    if (found):
        finalPath.append(actions[currentState])
        parent = pathMap[currentState]
        while (parent != problem.getStartState()):
            finalPath.append(actions[parent])
            currentState = parent
            parent = pathMap[currentState]
    return finalPath[::-1]

def breadthFirstSearch(problem):
    """
    Do a breadth first search
    """
    visited = []
    actions = dict()
    pathMap = dict()
    queue = Queue()
    queue.push(problem.getStartState())
    found = False
    currentState = None
    while (not queue.isEmpty() and not found):
        currentState = queue.pop()
        if (problem.isGoalState(currentState)):
            found = True
        else:
            if (currentState not in visited):
                visited.append(currentState)
                nextStates = problem.getSuccessors(currentState)
                for state in nextStates:
                    if (state[0] not in visited):
                        queue.push(state[0])
                        if (state[0] not in pathMap):
                            actions[state[0]] = state[1]
                            pathMap[state[0]] = currentState               
    finalPath = []
    " Form the final path if found "
    if (found):
        finalPath.append(actions[currentState])
        parent = pathMap[currentState]
        while (parent != problem.getStartState()):
            finalPath.append(actions[parent])
            currentState = parent
            parent = pathMap[currentState]
    return finalPath[::-1]

def uniformCostSearch(problem):
    "Search for least total cost first. "
    currentNode = Node(None, problem.getStartState(), None, 0)
    frontier = PriorityQueueWithFunction(lambda node: node.getTotalCost())
    frontier.push(currentNode)
    visited = []
    while (True):
        if (frontier.isEmpty()):
            return []
        currentNode = frontier.pop()
        " If we are at the goal state, form the path from previous nodes"
        if (problem.isGoalState(currentNode.getState())):
            solution = []
            while(currentNode.getPreviousNode() != None):
                solution.append(currentNode.getAction())
                currentNode = currentNode.getPreviousNode()
            return solution[::-1]
        if (currentNode.getState() not in visited):
            successors = problem.getSuccessors(currentNode.getState())
            for state in successors:
                newNode = Node(currentNode, *state)
                if (newNode.getState() not in visited):
                    frontier.push(newNode)
            visited.append(currentNode.getState())  

def aStarSearch(problem, heuristic=nullHeuristic):
    
    start = problem.getStartState()
    start = (start, None, 0)
    closed = []
    open = PriorityQueue()
    open.push(start, 0)
    iterableOpen = [start]
    
    expanded = []
    path = dict()
    
    " Dictionary for the step costs "
    g_map = dict()
    g_map[start] = 0
    
    " Dictionary for the total cost (step + estimated cost to goal)"
    f_map = dict()
    f_map[start] = g_map[start] + heuristic(start[0], problem)
    
    while (not open.isEmpty()):
        current = open.pop()
        iterableOpen.remove(current)
        "Form the path if we have reached the goal state"
        if (problem.isGoalState(current[0])):
            finalPath = []
            finalPath.append(current[1])
            parent = path[current]
            while (parent[0] != problem.getStartState()):
                finalPath.append(parent[1])
                current = parent
                parent = path[current]
            return finalPath[::-1]
        
        closed.append(current[0])
        if (current[0] not in expanded):
            expanded.append(current[0])
            for neighbor in problem.getSuccessors(current[0]):
                if (neighbor[0] not in closed):
                    temp_g = g_map[current] + neighbor[2]
                    
                    if neighbor not in iterableOpen or temp_g < g_map[neighbor]:
                        path[neighbor] = current
                        g_map[neighbor] = temp_g
                        f_map[neighbor] = g_map[neighbor] + heuristic(neighbor[0], problem)
                        if (neighbor not in iterableOpen):
                            open.push(neighbor, f_map[neighbor])
                            iterableOpen.append(neighbor)
    return ['fail']

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
