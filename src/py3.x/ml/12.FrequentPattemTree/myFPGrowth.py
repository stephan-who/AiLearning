print(__doc__)

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
            #    ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict.keys():
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict

def updateHeader(nodeToTest, targetNode):

    while(nodeToTest.nodeLink is not None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def updateTree(items, inTree, headerTable, count):
    """
    更新， 第二次遍历
    :param items:满足minSup排序后的元素key的数组（大到小的排序）
    :param inTree:空的Tree对象
    :param headerTable:满足minSup{所有的元素+(value, treeNode)}
    :param count:原数据集中每一组key出现的次数
    :return:
    """
    """
    取出元素出现次数最高的，如果该元素在inTree.children这个字典中，就进行累加；如果该元素不存在就inTree.childern 字典中新增key，value为treeNode对象
    """
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

def createTree(dataSet, minSup=1):

    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        # dist{元素key: [元素次数, None]}
        headerTable[k] = [headerTable[k], None]

    # create tree
    retTree = treeNode('Null Set', 1, None)
    # 循环dist {行， 出现次数}的样本数据
    for tranSet, count in dataSet.items():
        # localD = dist{元素key: 元素总出现次数}
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # p=key, value; 所以通过value值的大小，进行从大到小进行排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p : p[1], reverse=True)]
            # 填充树，通过有序的orderedItems的第一位，进行顺序填充，第一层的子节点
            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable

def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath)> 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    创建条件FP🌲
    :param inTree:
    :param headerTable:
    :param minSup:
    :param preFix:
    :param freqItemList:
    :return:
    """
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]
    print('------', sorted(headerTable.items(), key=lambda p:p[1][0]))
    print('bigL=', bigL)
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('newFreqSet=', newFreqSet, preFix)

        freqItemList.append(newFreqSet)
        print('freqItemList=', freqItemList)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('candPattBases=', basePat, condPattBases)

        # 构建条件 FP Tree
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            myCondTree.disp(1)
            print('\n\n\n')
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
        print('\n\n\n')

if __name__ == '__main__':
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    print(initSet)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    print(myHeaderTab)
    print('x --->', findPrefixPath('x', myHeaderTab['x'][1]))
    print('z --->', findPrefixPath('z', myHeaderTab['z'][1]))
    print('t --->', findPrefixPath('t', myHeaderTab['t'][1]))

    freqItemList = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItemList)
    print('freqItemList:\n', freqItemList)

    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPtree, myHeaderTab = createTree(initSet, 100000)

    myFreList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreList)
    print(myFreList)