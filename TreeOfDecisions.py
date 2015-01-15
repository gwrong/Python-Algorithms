# Project work from my intro AI class
# Did not include supporting files/most comments to preserve integrity

class Node:
  """
  A simple node class to build our tree with. It has the following:
  """
  
  def __init__(self,attr):
    self.children = {}
    self.attr = attr
    self.isleaf = False

class LeafNode(Node):
    """
    A basic extension of the Node class with just a value.
    """
    def __init__(self,value):
        self.value = value
        self.isleaf = True
    
class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  """
  def __init__(self, root=None):
    self.root = root

  def prettyPrint(self):
    print str(self)
    
  def preorder(self,depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if node.isleaf:
      return '|---'*depth+str(node.value)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.attr),str(val))
      string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
    return string   

  def count(self,node=None):
    if node is None:
      node = self.root
    if node.isleaf:
      return 1
    count = 1
    for child in node.children.values():
      if child is not None:
        count+= self.count(child)
    return count  

  def __str__(self):
    return self.preorder(0, self.root)
  
  def classify(self, classificationData):
    """
    Uses the classification tree with the passed in classificationData.`
    """
    node = self.root
    while (not isinstance(node, LeafNode)):
        node = node.children[classificationData[node.attr]]
    return node.value
  
def getPertinentExamples(examples,attrName,attrValue):
    """
    That is, this gets the list of examples that have the value 
    attrValue for the attribute with the name attrName.
    """
    newExamples = []
    for aDict in examples:
        if (aDict[attrName] == attrValue):
            newExamples.append(aDict)
    return newExamples
  
def getClassCounts(examples,className):
    """
    Helper function to get a list of counts of different class values
    in a set of examples. 
    """
    classCounts = {}
    for aDict in examples:
        classCounts[aDict[className]] = classCounts.get(aDict[className], 0) + 1
    return classCounts

def getMostCommonClass(examples,className):
    """
    Gets the most common class
    in the examples. See parameters in getClassCounts.
    """
    counts = getClassCounts(examples,className)
    return max(counts, key=counts.get) if len(examples)>0 else None

def getAttributeCounts(examples,attrName,attrValues,className):
    """
    Helper function to get a list of counts of different class values
    corresponding to every possible assignment of the passed in attribute. 
    """
    counts = []
    for attrValue in attrValues:
        classCount = dict()
        for aDict in examples:
            if(aDict[attrName] == attrValue):
               classCount[aDict[className]] = classCount.get(aDict[className], 0) + 1 
        counts.append(list(classCount.values()))
    return counts
        

def setEntropy(classCounts):
    """
    Calculates the set entropy value for the given list of class counts.
    """
    total = 0
    for value in classCounts:
        total += value
    result = 0
    for value in classCounts:
        ratio = float(value) / total
        result = result - ratio * log(ratio, 2)
    return result
   

def remainder(examples,attrName,attrValues,className):
    """
    Calculates the remainder value for given attribute and set of examples.
    """
    attrCounts = getAttributeCounts(examples, attrName, attrValues, className)
    classCounts = getClassCounts(examples, className)
    result = 0
    totalClass = 0
    for value in classCounts.values():
        totalClass += value
    for attrCount in attrCounts:
        numHere = 0
        for count in attrCount:
            numHere += count
        bigRatio = float(numHere) / totalClass
        iFunction = 0
        for value in attrCount:
            if (numHere > 0):
                ratio = float(value) / numHere
                iFunction = iFunction - ratio * log(ratio, 2)
        result = result + bigRatio * iFunction
    return result
          
def infoGain(examples,attrName,attrValues,className):
    """
    Calculates the info gain value for given attribute and set of examples.
    """
    classCounts = getClassCounts(examples, className)
    result = 0
    totalClass = 0
    for value in classCounts.values():
        totalClass += value
    for value in classCounts.values():
        iFunction = 0
        ratio = float(value) / totalClass
        iFunction = iFunction - ratio * log(ratio, 2)
        result = result + iFunction
    return result - remainder(examples, attrName, attrValues, className)
  
def giniIndex(classCounts):
    """
    Calculates the gini value for the given list of class counts.
    """
    total = 0
    result = 0
    for thing in classCounts:
        total += thing
    for thing in classCounts:
        result += (float(thing) / total) ** 2
    return 1 - result
  
def giniGain(examples,attrName,attrValues,className):
    """
    Return the inverse of the giniD function described in the instructions.
    """
    attrCounts = getAttributeCounts(examples, attrName, attrValues, className)
    classCounts = getClassCounts(examples, className)
    result = 0
    total = 0
    for thing in classCounts.values():
        total += thing
    for attrCount in attrCounts:
        currentTotal = 0
        for thing in attrCount:
            currentTotal += thing
        result += (float(currentTotal) / total) * giniIndex(attrCount)
    return 1 / float(result) if result != 0 else maxint

    
def makeTree(examples, attrValues,className,setScoreFunc,gainFunc):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    """
    remainingAttributes=attrValues.keys()
    return Tree(makeSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples, className),setScoreFunc,gainFunc))
    
def makeSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters.
    """
    if (not examples):
       return LeafNode(defaultLabel)
    elif (len(getClassCounts(examples, className).values()) == 1):
        return LeafNode(getMostCommonClass(examples, className))
    elif (not remainingAttributes):
        return LeafNode(getMostCommonClass(examples, className))
    else:
        bestAttr = None
        bestGain = float("-inf")
        bestAttr = max(remainingAttributes, key=lambda x:gainFunc(examples, x, attributeValues[x], className))
        
        node = Node(bestAttr)
        newRemAttr = []
        for item in remainingAttributes:
            if (item != bestAttr):
                newRemAttr.append(item)
        for attributeValue in attributeValues[bestAttr]:
            exs = getPertinentExamples(examples, bestAttr, attributeValue)
            node.children[attributeValue] = makeSubtrees(newRemAttr,exs,attributeValues,className,getMostCommonClass(examples, className),setScoreFunc,gainFunc)
        return node