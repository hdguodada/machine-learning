from numpy import *
import operator


def createDateSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    #数据的一维长度
    dataSetSize = dataSet.shape[0]
    #  1.若reps为一个数字n，则构造一个重复n次的一维的A‘

    #  2.若reps为一个元组(m,n)，则构造一个m行n列的矩阵，其中每个元素均为A，这样就形成了矩阵
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    # 输入向量点与已知数据各点的距离
    distances = sqDistance**0.5
    # sortedDistIndicies 是距离最近从小到大排序的索引值的列表
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        # 距离最近的点的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # 类别数+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 获得距离最近的类别
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    # 文件行数
    numberOfLines = len(arrayOlines)
    # 创建以0填充的矩阵。
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # 每行有4个数据
        line = line.strip()
        # 4数据组成的list
        listFromLine = line.split('\t')
        # 将数据转化为矩阵
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.05
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        a = normMat[i,:]
        b = normMat[numTestVecs:m,:]
        c = datingLabels[numTestVecs:m]
        classifierResult = classify0(a, b, c, 3)
        print('the classifier came back with : %d, the real answer is : %d'%(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
            print('the total error rate is :%f'%(errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('you will probably like this person:', resultList[classifierResult-1])


if __name__ == '__main__':
    classifyPerson()
