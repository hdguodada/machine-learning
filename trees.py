from math import log


def calcShanonEnt(dataSet):
    """
    计算熵值
    :param dataSet:
    :return:
    """
    # 实例总数
    numEntries = len(dataSet)
    # 数据字典,为记录每个类别出现的次数
    labelCouts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCouts.keys():
            labelCouts[currentLabel] = 0
        labelCouts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCouts:
        prob = float(labelCouts[key]/numEntries)
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


def createDataSet():
    """
    测试数据
    :return:
    """
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no'],
    ]
    labels = [
        'no surfacing', 'flippers'
    ]
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征值划分数据集
    :param dataSet:  待划分的数据集
    :param axis: 待划分数据的特征
    :param value:需要返回的特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


if __name__ == '__main__':
    mydat, labels = createDataSet()
    print(splitDataSet(mydat,0,1))
    print(splitDataSet(mydat,0,0))