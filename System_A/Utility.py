import math
import numpy as np


class Utility():
    def __init__(self):
        pass

    def countIncorecct(self, list):
        count = 0
        for i in range(len(list)):
            if list[i][1] != list[i][3]:
                count += 1
        print(count)

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
        print(dScore)
