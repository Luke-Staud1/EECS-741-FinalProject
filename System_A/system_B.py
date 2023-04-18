import cv2
import numpy as np
from Utility import Utility as util


class faceDetect():
    gallarySet = []
    probeSet = []
    gallarySetData = []
    probeSetData = []
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
        pass

    def readInImages(self):
        path1 = '../GallerySet/'
        path2 = '../ProbeSet/'
        setsize = 100
        for i in range(1, setsize):
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
        for i in range(len(self.probeSet)):
            tempProbe = np.split(self.probeSet[i], 5, axis=1)
            tempGallary = np.split(self.gallarySet[i], 5, axis=1)
            self.probeSetData.append(tempProbe)
            self.gallarySetData.append(tempGallary)

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
    faceDetect().upsacleSet(4)
    faceDetect().displayImages()
