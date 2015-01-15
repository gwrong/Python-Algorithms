# Project work from my intro AI class
# Did not include supporting files/most comments to preserve integrity

class UnaryConstraint:
	def __init__(self, var):
		self.var = var

	def isSatisfied(self, value):
		util.raiseNotDefined()

	def affects(self, var):
		return var == self.var

class BadValueConstraint(UnaryConstraint):
	def __init__(self, var, badValue):
		self.var = var
		self.badValue = badValue

	def isSatisfied(self, value):
		return not value == self.badValue

	def __repr__(self):
		return 'BadValueConstraint (%s) {badValue: %s}' % (str(self.var), str(self.badValue))

class GoodValueConstraint(UnaryConstraint):
	def __init__(self, var, goodValue):
		self.var = var
		self.goodValue = goodValue

	def isSatisfied(self, value):
		return value == self.goodValue

	def __repr__(self):
		return 'GoodValueConstraint (%s) {goodValue: %s}' % (str(self.var), str(self.goodValue))


"""
	Base class for binary constraints
	Implement isSatisfied in subclass to use
"""
class BinaryConstraint:
	def __init__(self, var1, var2):
		self.var1 = var1
		self.var2 = var2

	def isSatisfied(self, value1, value2):
		util.raiseNotDefined()

	def affects(self, var):
		return var == self.var1 or var == self.var2

	def otherVariable(self, var):
		if var == self.var1:
			return self.var2
		return self.var1


"""
	Implementation of BinaryConstraint
	Satisfied if both values assigned are different
"""
class NotEqualConstraint(BinaryConstraint):
	def isSatisfied(self, value1, value2):
		if value1 == value2:
			return False
		return True

	def __repr__(self):
		return 'NotEqualConstraint (%s, %s)' % (str(self.var1), str(self.var2))

class EqualConstraint(BinaryConstraint):
	def isSatisfied(self, value1, value2):
		if value1 == value2 or value1 == None or value2 == None:
			return True
		return False

	def __repr__(self):
		return 'EqualConstraint (%s, %s)' % (str(self.var1), str(self.var2))

class ConstraintSatisfactionProblem:
	
	def __init__(self, variables, domains, binaryConstraints = [], unaryConstraints = []):
		self.varDomains = {}
		for i in xrange(len(variables)):
			self.varDomains[variables[i]] = domains[i]
		self.binaryConstraints = binaryConstraints
		self.unaryConstraints = unaryConstraints
		

	def __repr__(self):
	    return '---Variable Domains\n%s---Binary Constraints\n%s---Unary Constraints\n%s' % ( \
	        ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
	        ''.join([str(e) + '\n' for e in self.binaryConstraints]), \
	        ''.join([str(e) + '\n' for e in self.unaryConstraints]))


class Assignment:
	
	def __init__(self, csp):
		self.varDomains = {}
		for var in csp.varDomains:
			self.varDomains[var] = set(csp.varDomains[var])
		self.assignedValues = { var: None for var in self.varDomains }

	
	def isAssigned(self, var):
		return self.assignedValues[var] != None

	
	def isComplete(self):
		for var in self.assignedValues:
			if not self.isAssigned(var):
				return False
		return True

	def extractSolution(self):
		if not self.isComplete():
			return None
		return self.assignedValues

	def __repr__(self):
	    return '---Variable Domains\n%s---Assigned Values\n%s' % ( \
	        ''.join([str(e) + ':' + str(self.varDomains[e]) + '\n' for e in self.varDomains]), \
	        ''.join([str(e) + ':' + str(self.assignedValues[e]) + '\n' for e in self.assignedValues]))



####################################################################################################


"""
	Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
	Do not assign value to var. Only check if this value would be consistent or not.
"""
def consistent(assignment, csp, var, value):
	binaryConstraints = csp.binaryConstraints
	for constraint in binaryConstraints:
		if (constraint.affects(var)):
			if (not constraint.isSatisfied(value, assignment.assignedValues[constraint.otherVariable(var)])):
				return False
	return True


"""
	Recursive backtracking algorithm.
"""
def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
	if (assignment.isComplete()):
		return assignment
	var = selectVariableMethod(assignment, csp)
	assignedValues = assignment.assignedValues
	inferences = None
	for value in orderValuesMethod(assignment, csp, var):
		if (consistent(assignment, csp, var, value)):
			assignment.assignedValues[var] = value
			inferences = inferenceMethod(assignment, csp, var, value)
			if inferences != None:
				#for inference in inferences:
					#assignedValues[inference[0]] = inference[1]
				result = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
				if (result != None):
					return result
		assignment.assignedValues[var] = None
		if (inferences != None):
			for inference in inferences:
				assignment.varDomains[inference[0]].add(inference[1])
	return None


"""
	Uses unary constraints to eliminate values from an assignment.
"""
def eliminateUnaryConstraints(assignment, csp):
	domains = assignment.varDomains
	for var in domains:
		for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
			for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
				domains[var].remove(value)
				if len(domains[var]) == 0:
					# Failure due to invalid assignment
					return None
	return assignment


"""
	Trivial method for choosing the next variable to assign.
"""
def chooseFirstVariable(assignment, csp):
	for var in csp.varDomains:
		if not assignment.isAssigned(var):
			return var

"""
	Selects the next variable to try to give a value to in an assignment with MRV/degree heuristic.
"""
def minimumRemainingValuesHeuristic(assignment, csp):
	
	nextVar = None
	binaryConstraints = csp.binaryConstraints
	domains = assignment.varDomains
	assignedValues = assignment.assignedValues
	minNum = float("inf")
	minVars = []
	for var in domains.keys():
		if (assignedValues[var] == None):
			count = len(domains[var])
			if (count == minNum):
				minVars.append(var)
			elif (count < minNum):
				minVars = [var]
				minNum = count
			
	if (len(minVars) > 1):
		maxNum = float("-inf")
		maxVar = minVars[0]
		for var in minVars:
			count = 0
			for constraint in binaryConstraints:
				if (constraint.affects(var)):
					count = count + 1
			if (count > maxNum):
				maxNum = count
				maxVar = var
		return maxVar
	
	return minVars[0]


"""
	Trivial method for ordering values to assign.
	Uses no heuristics.
"""
def orderValues(assignment, csp, var):
	return list(assignment.varDomains[var])


"""
	Creates an ordered list of the remaining values left for a given variable.
"""
def leastConstrainingValuesHeuristic(assignment, csp, var):
	values = list(assignment.varDomains[var])
	values.sort(key=lambda x: numConstrainedValues(assignment, csp, var, x))
	return values

def numConstrainedValues(assignment, csp, var, value):
	varDomains = assignment.varDomains
	binaryConstraints = csp.binaryConstraints
	count = 0
	for constraint in binaryConstraints:
		if constraint.affects(var):
			otherVar = constraint.otherVariable(var)
			remainingValues = varDomains[otherVar]
			for remainingValue in remainingValues:
				if (not constraint.isSatisfied(remainingValue, value)):
					count = count + 1
	return count

"""
	Trivial method for making no inferences.
"""
def noInferences(assignment, csp, var, value):
	return set([])


"""
	Implements the forward checking algorithm.
"""
def forwardChecking(assignment, csp, var, value):
	inferences = set([])
	domains = assignment.varDomains
	assignedValues = assignment.assignedValues
	binaryConstraints = csp.binaryConstraints
	for constraint in binaryConstraints:
		if (constraint.affects(var)):
			otherVariable = constraint.otherVariable(var)
			if (assignedValues[otherVariable] == None):
				otherVarDomain = domains[otherVariable]
				for otherDomainValue in list(otherVarDomain):
					if (not constraint.isSatisfied(value, otherDomainValue)):
						inferences.add((otherVariable, otherDomainValue))
						domains[otherVariable].remove(otherDomainValue)
						if (not domains[otherVariable]):
							for inference in inferences:
								domains[inference[0]].add(inference[1])
							return None
	
	return inferences


"""
	Helper function to maintainArcConsistency and AC3.
"""
def revise(assignment, csp, var1, var2, constraint):
	inferences = set([])
	varDomains = assignment.varDomains
	var1Domain = varDomains[var1]
	var2Domain = varDomains[var2]
	for value in list(var2Domain):
		found = False
		for anotherValue in list(var1Domain):
			if (constraint.isSatisfied(value, anotherValue)):
				found = True
		if (not found):
			inferences.add((var2, value))
			var2Domain.remove(value)
			if (not var2Domain):
				for inference in inferences:
					var2Domain.add(inference[1])
				return None
	return inferences


"""
	Implements the maintaining arc consistency algorithm.
"""
def maintainArcConsistency(assignment, csp, var, value):
	inferences = set([])
	queue = set()
	constraints = csp.binaryConstraints
	assignedValues = assignment.assignedValues
	for constraint in constraints:
		if (constraint.affects(var) and assignedValues[constraint.otherVariable(var)] == None):
			queue.add((var, constraint))
		
	while (queue):
		current = queue.pop()
		var = current[0]
		currentConstraint = current[1]
		result = revise(assignment, csp, var, currentConstraint.otherVariable(var), currentConstraint)
		if (result == None):
			for inference in inferences:
				assignment.varDomains[inference[0]].add(inference[1])
			return None
		else:
			for thing in result:
				inferences.add(thing)
				for constraint in constraints:
					if (constraint.affects(thing[0])):
						queue.add((thing[0], constraint))
	return inferences


"""
	AC3 algorithm for constraint propogation. Used as a preprocessing step to reduce the problem
	before running recursive backtracking.
"""
def AC3(assignment, csp):
	inferences = set([])
	queue = set()
	constraints = csp.binaryConstraints
	
	assignedValues = assignment.assignedValues
	varDomains = assignment.varDomains
	for constraint in constraints:
		queue.add((constraint.var1, constraint))
		queue.add((constraint.var2, constraint))
	while (queue):
		current = queue.pop()
		var = current[0]
		currentConstraint = current[1]
		result = revise(assignment, csp, var, currentConstraint.otherVariable(var), currentConstraint)
		if (result == None):
			return None
		else:
			for thing in result:
				for constraint in constraints:
					if (constraint.affects(thing[0])):
						queue.add((thing[0], constraint))
	return assignment