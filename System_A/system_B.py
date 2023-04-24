import cv2
import numpy as np
from Utility import Utility as util
import sys
import math
# from scipy.stats import mannwhitneyu
np.set_printoptions(threshold=sys.maxsize)


class faceDetect():
    gallarySet = []
    probeSet = []
    gallaryMask = []
    probeMask = []
    setSize = 100
    divideFactor = 5

    utility = util
    kernelBlur = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
    kernelDetect = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])

    def __init__(self):
        self.kernelAverage = np.ones((self.divideFactor, self.divideFactor))

    def readInImages(self):
        path1 = '../GallerySet/'
        path2 = '../ProbeSet/'
        for i in range(1, self.setSize+1):
            self.gallarySet.append(cv2.imread(path1 + 'subject' +
                                              str(i) + '_img1.pgm', cv2.COLOR_BGR2GRAY))
            self.probeSet.append(cv2.imread(path2 + 'subject' +
                                            str(i) + '_img2.pgm', cv2.COLOR_BGR2GRAY))

    def imageAverage(self, image):
        sum = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                sum += image[i][j]
        return int(sum / (image.shape[0] * image.shape[1]))

    def lookForFeatures(self):
        newGallarySet = np.zeros(np.array(self.gallarySet).shape)
        newProbeSet = np.zeros(np.array(self.probeSet).shape)
        for z in range(len(self.gallarySet)):

            # self.gallarySet[z] = util.convolveImage(
            #     util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)
            # self.gallarySet[z] = util.convolveImage(
            #     util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)
            # self.gallarySet[z] = util.convolveImage(
            #     util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)
            # self.gallarySet[z] = util.convolveImage(
            #     util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)

            thresh = self.imageAverage(self.gallarySet[z])
            newIm = [[]]
            for i in range(self.gallarySet[z].shape[0]):
                temp = []
                for j in range(self.gallarySet[z].shape[1]):
                    if(self.gallarySet[z][i][j] > thresh):
                        newGallarySet[z][i][j] = 1
                    else:
                        newGallarySet[z][i][j] = 0
            # newGallarySet[z] = util.erode(util, newGallarySet[z])
            # newGallarySet[z] = util.dialate(util, newGallarySet[z])

        for z in range(len(self.probeSet)):
            # self.probeSet[z] = util.convolveImage(
            #     util, self.probeSet[z], self.kernelBlur, 1.0 / 256)
            # self.probeSet[z] = util.convolveImage(
            #     util, self.probeSet[z], self.kernelBlur, 1.0 / 256)
            # self.probeSet[z] = util.convolveImage(
            #     util, self.probeSet[z], self.kernelBlur, 1.0 / 256)
            # self.probeSet[z] = util.convolveImage(
            #     util, self.probeSet[z], self.kernelBlur, 1.0 / 256)

            thresh = self.imageAverage(self.probeSet[z])
            newIm = [[]]
            for i in range(self.probeSet[z].shape[0]):
                temp = []
                for j in range(self.probeSet[z].shape[1]):
                    if(self.probeSet[z][i][j] > thresh):
                        newProbeSet[z][i][j] = 1
                    else:
                        newProbeSet[z][i][j] = 0
            # newProbeSet[z] = util.erode(util, newProbeSet[z])
            # newProbeSet[z] = util.dialate(util, newProbeSet[z])
        self.gallaryMask = newGallarySet
        self.probeMask = newProbeSet

    def HammingDistance(self, image1, image2):
        if len(image1) == len(image2) and len(image1[0]) == len(image2[0]):
            difference = 0
            for row1, row2 in zip(image1, image2):
                for pixel1, pixel2 in zip(row1, row2):
                    if pixel1 != pixel2:
                        difference += 1

        return difference

    def upscale(self, image, amount):
        newimage = cv2.resize(image, (0, 0), fx=amount, fy=amount)
        return newimage

    def upsacleSet(self, amount):
        for i in range(len(self.gallarySet)):
            self.gallarySet[i] = self.upscale(self.gallarySet[i], amount)
            self.probeSet[i] = self.upscale(self.probeSet[i], amount)

    def displayImages(self):
        # for i in range(len(self.gallarySet)):
        for i in range(10):
            cv2.imshow('Gallary Set', self.gallarySet[i])
            cv2.imshow('Probe Set', self.probeSet[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def splitAndAverage(self, image):
        output = np.zeros((math.ceil(image.shape[0]/self.divideFactor),
                           math.ceil(image.shape[0]/self.divideFactor)))
        size = self.kernelAverage.shape[0]
        sIndex = int(math.floor(size/2))
        # output = np.zeros(np.array(image).shape/averageRange)
        temp = util.convolveImage(util, np.array(
            image), self.kernelAverage, 1.0)
        for i in range(sIndex, image.shape[0], size):
            for j in range(2, image.shape[1], size):
                i_o = int((i-sIndex)/size)
                j_o = int((j-sIndex)/size)
                output[i_o][j_o] = temp[i][j]
        return(output)

    def computeDifference(self, image1, image2):
        difference = 0
        for row1, row2 in zip(image1, image2):
            for pixel1, pixel2 in zip(row1, row2):
                difference += abs(pixel1 - pixel2)
        return difference

    def run(self):
        self.readInImages()
        self.lookForFeatures()
        upperLimit = 100
        score_matrix = []
        score_matrix2 = []
        for i in range(len(self.probeMask)):
            temp = []
            for j in range(len(self.gallaryMask)):
                score = self.HammingDistance(
                    self.probeMask[i], self.gallaryMask[j])
                temp.append(score)
            print(i)
            score_matrix.append(temp)

        for i in range(len(self.probeMask)):
            temp = []
            for j in range(len(self.gallaryMask)):
                matrix1 = self.splitAndAverage(self.probeMask[i])
                matrix2 = self.splitAndAverage(self.gallaryMask[j])
                score = self.computeDifference(matrix1, matrix2)
                temp.append(score)
            print(i)
            score_matrix2.append(temp)
        # print(score_matrix2)
        util.countIncorrect(util, score_matrix)
        util.countIncorrect(util, score_matrix2)
        util.computeDScore(util, score_matrix)
        util.computeDScore(util, score_matrix2)


if __name__ == "__main__":
    faceDetect().run()
