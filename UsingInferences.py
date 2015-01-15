# Project work from my intro AI class
# Did not include supporting files/most comments to preserve integrity

class InferenceModule:
    
    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the given gameState.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the specified
        position in the supplied gameState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)


    def initializeUniformly(self, gameState):
        pass

    def observe(self, observation, gameState):
        pass

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm
    updates to compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        """
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        beliefs = self.beliefs
        legalPositions = self.legalPositions
        
        allPossible = util.Counter()
        if (noisyDistance == None):
            allPossible[self.getJailPosition()] = 1
        else:
            for p in self.legalPositions:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                if emissionModel[trueDistance] > 0: 
                    allPossible[p] = (emissionModel[trueDistance] * beliefs[p])

        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        """
        Update self.beliefs in response to a time step passing from the current state.
        """

        beliefs = self.beliefs
        newBeliefs = util.Counter()
        for oldPos in beliefs.keys():
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            for newPos, prob in newPosDist.items():
                newBeliefs[newPos] = newBeliefs[newPos] +  prob * beliefs[oldPos]
        newBeliefs.normalize()
        self.beliefs = newBeliefs
        
    def getBeliefDistribution(self):
        return self.beliefs

class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """


    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        """
          Initializes a list of particles.
        """
        numParticles = self.numParticles
        legalPositions = self.legalPositions
        particles = []
        each = numParticles / len(legalPositions)
        for position in legalPositions:
            for x in range(0, each):
                particles.append(position)
        self.particles = particles
        
    def observe(self, observation, gameState):
        """
        Update beliefs based on the given distance observation. 
        """

        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        if (noisyDistance == None):
            newParticles = []
            for particle in self.particles:
                newParticles.append(self.getJailPosition())
            self.particles = newParticles
            return
        
        #For a particle filter, the belief distribution is over ONLY
        #the list of current particles
        beliefs = self.getBeliefDistribution()
        
        newBeliefs = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0: 
                newBeliefs[p] = (emissionModel[trueDistance] * beliefs[p])
        if (newBeliefs.totalCount() == 0):
            self.initializeUniformly(gameState)
            newBeliefs = self.getBeliefDistribution()
        newParticles = []
        particles = self.particles
        for particle in particles:
            newParticles.append(util.sample(newBeliefs))
        self.particles = newParticles
        
        

    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.
        """
        #For a particle filter, the belief distribution is over ONLY
        #the list of particles
        beliefs = self.getBeliefDistribution()
        newBeliefs = util.Counter()
        for oldPos in beliefs.keys():
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            for newPos, prob in newPosDist.items():
                newBeliefs[newPos] = newBeliefs[newPos] +  prob * beliefs[oldPos]
        newBeliefs.normalize()
        newParticles = []
        particles = self.particles
        for particle in particles:
            newParticles.append(util.sample(newBeliefs))
        self.particles = newParticles

    def getBeliefDistribution(self):
        """
          Return the agent's current belief state, a distribution over
          ghost locations conditioned on all evidence and time passage.
        """
        #For a particle filter, the belief distribution is over ONLY
        #the list of particles
        beliefs = util.Counter()
        for particle in self.particles:
            beliefs[particle] = beliefs[particle]+ 1
        return util.normalize(beliefs)
class MarginalInference(InferenceModule):

    def initializeUniformly(self, gameState):
        if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        if self.index == 1: jointInference.observeState(gameState)

    def elapseTime(self, gameState):
        if self.index == 1: jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist

class JointParticleFilter:
    "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.  
        """
        
        numParticles = self.numParticles
        legalPositions = self.legalPositions
        permutations = legalPositions
        for x in range(self.numGhosts - 1):
            permutations = itertools.product(permutations, legalPositions)
        permutations = list(permutations)
        each = numParticles / len(permutations)
        particles = []
        for permute in permutations:
            for x in range(each):
                particles.append(permute)
        self.particles = particles

    def addGhostAgent(self, agent):
        "Each ghost agent is registered separately and stored (in case they are different)."
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observeState(self, gameState):
        """
        Resamples the set of particles using the likelihood of the noisy observations.

        """
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()
        if len(noisyDistances) < self.numGhosts: return
        emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]
        particles = self.particles
        newBeliefs = util.Counter()
        
        for particle in particles:
            #We have to keep this variable since we are dealing with multiple ghosts
            emissionTotal = 1
            for i in range(self.numGhosts):
                if (noisyDistances[i] == None):
                    particle = self.getParticleWithGhostInJail(particle, i)
                else:
                    trueDistance = util.manhattanDistance(particle[i], pacmanPosition)
                    emissionTotal = emissionTotal * emissionModels[i][trueDistance]
            '''For the particle, we looked at all ghosts in it, so go ahead and add it to the beliefs
               for that particle'''
            newBeliefs[particle] = newBeliefs[particle] + emissionTotal
        if (newBeliefs.totalCount() == 0):
            self.initializeParticles()
        else:
            #resample
            newParticles = []
            particles = self.particles
            for particle in particles:
                newParticles.append(util.sample(newBeliefs))
            self.particles = newParticles

    def getParticleWithGhostInJail(self, particle, ghostIndex):
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        """
        Samples each particle's next state based on its current state and the gameState.
        """
        beliefs = self.getBeliefDistribution()
        newParticles = []
        for particle in self.particles:
            newParticle= ()
            for i in range(self.numGhosts):
                beliefsByGhost = util.Counter()
                newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, particle), i, self.ghostAgents[i])
                for newPos, prob in newPosDist.items():
                    beliefsByGhost[newPos] = beliefsByGhost[newPos] + prob * beliefs[particle]
                #We are constructing the new particle as we look at each old particle since
                #the position distribution we get for each ghost is unique to that particle
                #Thus, we have to have a unique belief distribution for each ghost's next move
                #for each particle
                newParticle = newParticle + (util.sample(beliefsByGhost),)
            newParticles.append(newParticle)
                
        self.particles = newParticles
        
        

    def getBeliefDistribution(self):
        beliefs = util.Counter()
        for particle in self.particles:
            beliefs[particle] = beliefs[particle]+ 1
        return util.normalize(beliefs)

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied gameState.
    """
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState

