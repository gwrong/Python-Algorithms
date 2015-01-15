# Project work from my intro AI class
# Did not include supporting files/most comments to preserve integrity

class Perceptron(object):
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        """
        return 1 / (1 + e ** (-value))
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        """
        newInacts = [1.0] + inActs
        sum = self.getWeightedSum(newInacts)
        return round(self.sigmoid(sum))
        
    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        """
        return self.sigmoid(value) * (1 - self.sigmoid(value))
        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron
        """
        newInacts = [1] + inActs
        sum = self.getWeightedSum(newInacts)
        return self.sigmoidDeriv(sum)
    
    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        """
        totalModification = 0
        newInacts = [1] + inActs
        for x in range(len(self.weights)):
            newWeight = self.weights[x] + alpha * newInacts[x] * delta
            totalModification += abs(newWeight - self.weights[x])
            self.weights[x] = newWeight
        return totalModification
            
    def setRandomWeights(self):
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
    def __str__(self):
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        """
        result = []
        newInacts = list(inActs)
        result.append(newInacts)
        for layer in self.layers:
            subList = []
            for perceptron in layer:
                subList.append(perceptron.sigmoidActivation(newInacts))
            result.append(list(subList))
            newInacts = list(subList)
        return result
    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        """
        #keep track of output
        averageError = 0.0
        averageWeightChange = 0.0
        numWeights = 0
        errorSum = 0.0
        weightSum = 0.0
        perceptronSum = 0.0
        numPerceptrons = 0
        for layer in self.layers:
            for perceptron in layer:
                numPerceptrons = numPerceptrons + 1
        
        for example in examples:#for each example
            deltas = dict()#keep track of deltas to use in weight change
            expectedOutput = example[1]
            feedResult = self.feedForward(example[0])
            outputResult = feedResult[-1]
            index = 0
            outputInActs = feedResult[-2]
            for outputNode in self.outputLayer:
                delta = outputNode.sigmoidActivationDeriv(outputInActs) * (expectedOutput[index] - outputResult[index])
                deltas[outputNode] = delta
                errorSum += (expectedOutput[index] - outputResult[index]) ** 2 / 2
                index = index + 1
            for x in range(len(self.layers) - 2, -1, -1):
                index = x
                hiddenInActs = feedResult[index]
                weightIndex = 1
                for perceptron in self.layers[x]:
                    number = 0.0
                    
                    for otherPerceptron in self.layers[x + 1]:
                        number += otherPerceptron.weights[weightIndex] * deltas[otherPerceptron]
                    weightIndex = weightIndex + 1
                    delta = perceptron.sigmoidActivationDeriv(hiddenInActs) * number
                    deltas[perceptron] = delta
                index = index - 1
            feedIndex = 0
            for layer in self.layers:
                inActs = feedResult[feedIndex]
                for perceptron in layer:
                    weightSum += perceptron.updateWeights(inActs, alpha, deltas[perceptron])
                    perceptronSum += len(inActs) + 1 #For bias weights add 1
                feedIndex = feedIndex + 1
        averageError = errorSum / (len(examples) * len(outputResult))
        averageWeightChange = weightSum / (perceptronSum)
        return (averageError, averageWeightChange)
    
def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    """
    examplesTrain,examplesTest = examples
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet
    iteration=0
    trainError=0
    weightMod=1
    
    """
    Iterate for as long as it takes
    """
    while (weightMod > weightChangeThreshold and iteration < maxItr):
        trainError, weightMod = nnet.backPropLearning(examplesTrain, alpha)
        if iteration%10==0:
            print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        else :
            print '.',
        iteration= iteration + 1    
    
        
    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy
    """ 
    
    testError = 0
    testGood = 0
    for test in examplesTest:
        result = nnet.feedForward(test[0])[-1]
        if (result == test[1]):
            testGood = testGood + 1
        else:
            testError = testError + 1
    
    testAccuracy = float(testGood) / (testGood + testError)#num correct/num total
    
    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testGood,testError,testAccuracy)
    
    return nnet, testAccuracy