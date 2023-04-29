import math
import numpy as np


class Utility():
    def __init__(self):
        pass

    def countIncorecct(self, matrix):
        data = np.array(matrix)
        for i in range(data.shape[0]):
            temp = 10000000
            location = 0
            for j in range(data.shape[1]):
                if data[i, j] < temp:
                    temp = data[i, j]
                    location = j
            if(i != location):
                print("lowest: ", temp, " actual: ",
                      data[i, i], " difference: ", temp - data[i, i])

    def computeDScore(self, matrix):
        imposterList = []
        genuineList = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == j:
                    genuineList.append(matrix[i][j])
                else:
                    imposterList.append(matrix[i][j])
        IMean = np.mean(imposterList)
        GMean = np.mean(genuineList)
        IVar = np.var(imposterList)
        GVar = np.var(genuineList)
        dScore = (math.sqrt(2)*abs(IMean - GMean)) / \
            (math.sqrt(IVar + GVar))
        # print(dScore)
        return dScore

    def quadraticCurve(self, value, factor,):
        out = 0
        # print(factor+other)
        if value < (factor+67):
            out = value*0.1
        else:
            out = value
        # out = (value/2)
        return out

    def weightScore(self, matrix):

        # self.countIncorecct(matrix)
        original = np.array(matrix)
        lowest100 = np.zeros(original.shape[0])
        lowest100.fill(10000)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if (matrix[i][j] < max(lowest100)):
                    for k in range(len(lowest100)):
                        if matrix[i][j] < lowest100[k]:
                            lowest100 = np.delete(
                                lowest100, lowest100.shape[0]-1)
                            lowest100 = np.append(lowest100, matrix[i][j])
                            lowest100 = np.sort(lowest100)
                            break

        # print(max(lowest100))
        best = 0
        newMatrix = np.zeros(original.shape)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                newMatrix[i][j] = self.quadraticCurve(
                    matrix[i][j], max(lowest100))
        temp = self.computeDScore(newMatrix)
        print(temp)