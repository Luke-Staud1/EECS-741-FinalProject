import math
import numpy as np


class Utility():
    kernelDeNoise = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])

    def __init__(self):
        pass

    def countIncorrect(self, matrix):
        count = 0
        for i in range(len(matrix)):
            lowestVal = 100000000
            for j in range(len(matrix[i])):
                if matrix[i][j] < lowestVal:
                    lowestVal = matrix[i][j]
            if matrix[i][i] != lowestVal:
                count += 1
        print(count)
        return count

    def computeDScore(self, matrix):
        imposterList = []
        genuineList = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == j:
                    genuineList.append(matrix[i][j])
                else:
                    imposterList.append(matrix[i][j])
        # print(genuineList)
        IMean = np.mean(imposterList)
        GMean = np.mean(genuineList)
        IVar = np.var(imposterList)
        GVar = np.var(genuineList)
        dScore = (math.sqrt(2)*abs(IMean - GMean)) / \
            (math.sqrt(IVar + GVar))
        print(dScore)

    def convolveImage(self, image, kernel, factor=1):
        m, n = kernel.shape
        paddingAmount = int((m-1)/2)
        paddedImage = self.zeroPad(self, image, paddingAmount)
        newImage = np.zeros(image.shape)
        for row in range(1, image.shape[0]):
            for col in range(1, image.shape[1]):
                value = factor*np.sum(paddedImage[row:row+m, col:col+m]*kernel)
                if value > 255:
                    value = 255
                newImage[row][col] = value
        return newImage.astype(np.uint8)

    def zeroPad(self, image, padding=1):
        imagePadded = np.zeros(
            (image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding),
                    int(padding):int(-1 * padding)] = image

        return imagePadded.astype(np.uint8)

    def dialate(self, image):
        newImage = np.zeros(image.shape)
        tempImage = self.convolveImage(self, image, self.kernelDeNoise)
        for row in range(tempImage.shape[0]):
            for col in range(tempImage.shape[1]):
                if tempImage[row][col] > 0:
                    newImage[row][col] = 1
        return newImage

    def erode(self, image):
        newImage = np.zeros(image.shape)
        tempImage = self.convolveImage(self, image, self.kernelDeNoise)
        for row in range(tempImage.shape[0]):
            for col in range(tempImage.shape[1]):
                if tempImage[row][col] < 9:
                    newImage[row][col] = 0
                else:
                    newImage[row][col] = 1
        return newImage
