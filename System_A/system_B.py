import cv2
import numpy as np
from Utility import Utility as util
import sys
# from scipy.stats import mannwhitneyu
np.set_printoptions(threshold=sys.maxsize)


class faceDetect():
    gallarySet = []
    probeSet = []
    gallarySetData = []
    probeSetData = []
    setSize = 10
    divideFactor = 10

    utility = util
    kernelDeNoise = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
    kernelBlur = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
    kernelDetect = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])

    def __init__(self):
        self.gallarySetDataA = np.zeros(
            (self.setSize, self.divideFactor, self.divideFactor))
        self.probeSetDataA = np.zeros(
            (self.setSize, self.divideFactor, self.divideFactor))

    def readInImages(self):
        path1 = '../GallerySet/'
        path2 = '../ProbeSet/'
        for i in range(1, self.setSize+1):
            self.gallarySet.append(cv2.imread(path1 + 'subject' +
                                              str(i) + '_img1.pgm', cv2.COLOR_BGR2GRAY))
            self.probeSet.append(cv2.imread(path2 + 'subject' +
                                            str(i) + '_img2.pgm', cv2.COLOR_BGR2GRAY))

    def displayData(self):
        print(self.gallarySet[1].shape)

    def lookForFeatures(self):
        for z in range(len(self.gallarySet)):

            self.gallarySet[z] = util.convolveImage(
                util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)
            self.gallarySet[z] = util.convolveImage(
                util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)
            self.gallarySet[z] = util.convolveImage(
                util, self.gallarySet[z], self.kernelBlur, 1.0 / 256)
            for i in range(self.gallarySet[z].shape[0]):
                for j in range(self.gallarySet[z].shape[1]):
                    quantize = self.gallarySet[z][i][j] // 64
                    self.gallarySet[z][i][j] = quantize * 64

        for z in range(len(self.probeSet)):
            self.probeSet[z] = util.convolveImage(
                util, self.probeSet[z], self.kernelBlur, 1.0 / 256)
            self.probeSet[z] = util.convolveImage(
                util, self.probeSet[z], self.kernelBlur, 1.0 / 256)
            self.probeSet[z] = util.convolveImage(
                util, self.probeSet[z], self.kernelBlur, 1.0 / 256)
            for i in range(self.probeSet[z].shape[0]):
                for j in range(self.probeSet[z].shape[1]):
                    quantize = self.probeSet[z][i][j] // 64
                    self.probeSet[z][i][j] = quantize * 64

    def splitImage(self):
        gallary = np.array(self.gallarySet)
        probe = np.array(self.probeSet)
        tempProbe = []
        tempGallary = []
        probeAverage = []
        gallaryAverage = []
        setcount = int(50/self.divideFactor)
        for i in range(len(gallary)):
            for j in range(0, gallary[i].shape[0], setcount):
                for k in range(0, gallary[i].shape[1], setcount):
                    tempGallary.append(gallary[i][j:j+5, k:k+5])
                    tempProbe.append(probe[i][j:j+5, k:k+5])
        self.gallarySetData = np.array_split(tempGallary, self.setSize)
        self.probeSetData = np.array_split(tempProbe, self.setSize)
        print(np.array(self.gallarySetData).shape)
        for i in range(len(self.gallarySetData)):
            for j in range(len(self.gallarySetData[i])):
                gallaryAverage.append(np.average(self.gallarySetData[i][j]))
                probeAverage.append(np.average(self.probeSetData[i][j]))
        self.gallarySetDataA = np.array_split(gallaryAverage, self.setSize)
        self.ProbeSetDataA = np.array_split(probeAverage, self.setSize)
        print(self.gallarySetDataA[1])
        print(self.probeSetDataA[1])
        # print(self.probeSetData)

    def averageRegions(self):
        pass
        # print(self.gallarySetData)
        # print(self.probeSetData)

    def diffScore(self):
        # for i in range(len(self.gallarySetDataA)):
        # # print(self.gallarySetDataA)
        # print(self.gallarySetDataA)
        # print(self.probeSetDataA)

        # print(temp)

        pass

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


if __name__ == "__main__":
    faceDetect().readInImages()
    faceDetect().lookForFeatures()
    faceDetect().splitImage()
    # faceDetect().averageRegions()
    # faceDetect().diffScore()
    # faceDetect().upsacleSet(4)
    # faceDetect().displayImages()
