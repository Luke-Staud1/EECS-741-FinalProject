# Utility.py
import math
import numpy as np


class Utility():
    def __init__(self):
        pass

    def matrixToCSV(self, matrix):
        np.savetxt("score_matrix.csv", matrix, delimiter=",")  # save as csv

    def countIncorecct(self, matrix):  # count the number of incorrect matches
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

    def computeDScore(self, matrix):  # compute the d-score
        imposterList = []
        genuineList = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == j:
                    # add to genuine list if on diagonal
                    genuineList.append(matrix[i][j])
                else:
                    # add to imposter list otherwise
                    imposterList.append(matrix[i][j])
        IMean = np.mean(imposterList)
        GMean = np.mean(genuineList)
        IVar = np.var(imposterList)
        GVar = np.var(genuineList)
        print("imposter mean: ", IMean, " genuine mean: ", GMean,
              "Imposter Variance", IVar, "Genuine Variance", GVar)
        dScore = (math.sqrt(2)*abs(GMean - IMean)) / \
            (math.sqrt(IVar + GVar))  # compute d-score
        print(dScore)
        return dScore

    def MakeSnipit(self, matrix, size=10):  # make a snipit of the matrix for the report
        output = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                output[i][j] = matrix[i][j]
        print(output)
        self.matrixToCSV(output)

    def quantizedCurve(self, value, factor):  # quantize the curve using weighting curve
        out = 0
        # 67 is a tuned value the results the most accurate matching results for 1st and second best fits
        if value < (factor+67):
            out = value*0.1
        else:
            out = value
        return out

    # The fallowing code is the algorithm developed for identification
    def weightScore(self, matrix):
        original = np.array(matrix)
        lowest100 = np.zeros(original.shape[0])
        lowest100.fill(10000)  # fill with large number
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if (matrix[i][j] < max(lowest100)):
                    for k in range(len(lowest100)):
                        if matrix[i][j] < lowest100[k]:
                            lowest100 = np.delete(
                                lowest100, lowest100.shape[0]-1)  # remove last element
                            lowest100 = np.append(
                                lowest100, matrix[i][j])  # add new element
                            lowest100 = np.sort(lowest100)  # sort
                            break  # break out of loop
        newMatrix = np.zeros(original.shape)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                newMatrix[i][j] = self.quantizedCurve(
                    matrix[i][j], max(lowest100))
        temp = self.computeDScore(newMatrix)
        self.MakeSnipit(newMatrix, 10)
        print("D-Score: ", temp)
        return newMatrix
