import cv2  # For reading the image
import matplotlib.pyplot as plt  # for displaying Images / Plots
import copy
import sys
from Utility import Utility


def main():
    score_Data = []
    score_matrix = []

    upperLimit = 100
    for i in range(1, upperLimit):
        temp = []
        # Used for reading Tool Image in black and white
        imagePName = 'subject' + str(i) + '_img2.pgm'
        image2 = cv2.imread(
            '../ProbeSet/' + imagePName, cv2.COLOR_BGR2GRAY)
        closest = ['', 0, '', 0, sys.maxsize]
        for j in range(1, upperLimit):
            # Used for reading Tool Image in black and white
            imageGName = 'subject' + str(j) + '_img1.pgm'
            image1 = cv2.imread('../GallerySet/' +
                                imageGName, cv2.COLOR_BGR2GRAY)
            score = ComparePlots(image1, image2)
            temp.append(score)
        score_matrix.append(temp)
        print(i)
    Utility().computeDScore(score_matrix)


def Binarization(img, threshold):

    newimg = copy.deepcopy(img)
    for y in range(len(img)):
        for x in range(len(img[0])):
            if (img[y][x] > threshold):
                newimg[y][x] = 1
            else:
                newimg[y][x] = 0
    return newimg


def HammingDistance(image1, image2):
    if len(image1) == len(image2) and len(image1[0]) == len(image2[0]):
        difference = 0
        for row1, row2 in zip(image1, image2):
            for pixel1, pixel2 in zip(row1, row2):
                if pixel1 != pixel2:
                    difference += 1
        return difference


def ComparePlots(image1, image2):
    image1 = Binarization(image1, 128)
    image2 = Binarization(image2, 128)
    return HammingDistance(image1, image2)


def DisplayPlots(image1, image2):

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=.3, hspace=.3)

    plt.subplot(2, 2, 1)
    plt.title("Gallery Set")
    plt.imshow(image1, cmap=plt.cm.gray)

    plt.subplot(2, 2, 2)
    plt.title("Binary Image (Thresholding)")
    image1 = Binarization(image1, 128)
    plt.imshow(image1, cmap=plt.cm.gray)

    plt.subplot(2, 2, 3)
    plt.title("Probe Set")
    plt.imshow(image2, cmap=plt.cm.gray)

    plt.subplot(2, 2, 4)
    plt.title("Binary Image (Thresholding)")
    image2 = Binarization(image2, 128)
    plt.imshow(image2, cmap=plt.cm.gray)

    plt.show()


if __name__ == "__main__":
    main()
