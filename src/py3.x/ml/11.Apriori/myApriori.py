from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    """
    创建集合C1， 即对dataSet 进行去重，排序，放入list中，然后转换所有的元素为frozenSet
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 遍历所有的元素，如果不在C1出现过，那么就append
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)

def scanD(D, Ck, minSupport):
    """
    计算候选数据集CK 在数据集D中的支持度， 并放回支持度大于最小支持度（minSupport）的数据
    :param D:
    :param Ck:
    :param minSupport:
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 支持度 = 候选项（key） 出现的次数 / 所有数据集的数量
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 在retList的首位插入元素
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def apriorGen(Lk, k):
    """
    输入频繁相集列表Lk与返回的元素个数k， 然后输出所有可能的候选项集Ck
    :param Lk:
    :param k:
    :return:
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[: k-2] # 前k-2个元素
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    """
    首先构建集合C1， 然后扫描数据集来判断这些只有1个元素的项集是否满足最小支持度的要求。那么满足最小支持度要求的项集构成集合L1。然后L1中的元素相互组合成C2，C2再进一步过滤编程L2， 然后以此类推，直到CN的长度为0时结束，即可找出所有频繁项集的支持度。
    :param dataSet:
    :param minSupport:
    :return:
    """
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)

    L = [L1]
    k = 2
    # 判断L的第k-2项的数据长度是否>0. 第一次执行时L为[[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]. L[k-2] = L[0] = [frozenset([1]) frozenset([3]), frozenset([2]), frozenset([5])]],最后面k+=1
    while(len(L[k-2]) > 0):
        Ck = apriorGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        supportData.update(supK)
        if len(Lk) ==0:
            break
        L.append(Lk)
        k +=  1
    return L, supportData

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """

    :param freqSet: 频繁项集中的元素，例如: frozenset([2, 3, 5])
    :param H: 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
    :param supportData:
    :param brl: 关联规则列表的数组
    :param minConf:
    :return: prunedH 记录可信度大于阈值的集合
    """
    prunedH = []
    for conseq in H:
        """
        假设  freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，那么 frozenset([1]) 至 frozenset([3]) 的可信度为 = support(a | b) / support(a) = supportData[freqSet]/supportData[freqSet-conseq] = supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        """
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append(freqSet-conseq, conseq, conf)
            prunedH.append(conseq)
    return prunedH

#递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，
    长度由 aprioriGen(H, m+1) 这里的 m + 1 来控制
    :param freqSet:
    :param H:
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    """
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = apriorGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print('Hmp1=', Hmp1)
        print('len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet))
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    """

    :param L:
    :param supportData:
    :param minConf:
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    return bigRuleList
