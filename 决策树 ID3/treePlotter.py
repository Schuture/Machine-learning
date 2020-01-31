import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="1.8")
leafNode = dict(boxstyle="round4", fc="1.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    Input:  node content, 
            point coordinate to put content
            parent point x, y
            node type: bounding box type
    Output: a node with content and arrow in the plot
    '''
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
    
def getNumLeafs(myTree):
    '''
    Input:  tree denoted by a dict
    Output: number of leaves of the tree
    '''
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key]) # recursive
        else:   
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    '''
    Input:  tree denoted by a dict
    Output: max depth of the tree
    '''
    maxDepth = 0
    # firstStr = myTree.keys()[0]
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # test to see if the nodes are dictonaires, if not they are leaf nodes
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key]) # recursive
        else:   # leaf
            thisDepth = 1
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    '''
    compute the middle position of parent and child, add text here
    '''
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):# if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # the width of the plot
    depth = getTreeDepth(myTree)    # the height of the plot
    firstStr = list(myTree.keys())[0]     # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        #test to see if the nodes are dictonaires, if not they are leaf nodes  
        if type(secondDict[key]).__name__=='dict': 
            plotTree(secondDict[key],cntrPt,str(key))  #recursion
        else:   # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    

def createPlot(inTree):
    '''
    Input:  a tree denoted by a dict
    
    create plot area, calculate the figsize of tree, 
    call plotTree() recursively
    '''
    fig = plt.figure(1, figsize = (15,10), facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # no ticks(axis)
    #createPlot.ax1 = plt.subplot(111, frameon=False) # ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


def main():
    myTree = {'no surfacing': {0: 'no', 1:{ 'flippers': {0: "no", 1: 'yes'}}, 3: 'maybe'}}
    createPlot(myTree)


if __name__ == '__main__':
    main()





















