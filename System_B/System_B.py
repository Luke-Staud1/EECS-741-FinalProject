import cv2  # For reading the image
import matplotlib.pyplot as plt  # for displaying Images / Plots
from Utility import Utility
import numpy as np
import os


def main():
    probePATH = '../ProbeSet/'
    probeImg = loadGallery(probePATH)  # get np array of probe images
    galleryPATH = '../GallerySet/'
    galleryImg = loadGallery(galleryPATH)  # get np array of gallery images

    score_matrix = []
    upperLimit = 100

    for i in range(upperLimit):
        temp = []
        image2 = probeImg[i]
        for j in range(upperLimit):
            image1 = galleryImg[j]
            score = ComparePlots(image1, image2)
            temp.append(score)
        score_matrix.append(temp)
        print(i)
    Utility().weightScore(score_matrix)
    Utility().computeDScore(score_matrix)


def loadGallery(PATH):
    gallery = []
    for file in os.listdir(PATH):
        imgPATH = os.path.join(PATH, file)
        img = cv2.imread(imgPATH, cv2.IMREAD_GRAYSCALE)
        gallery.append(img.flatten())
    gallery = np.array(gallery)
    return gallery


def ComparePlots(image1, image2):
    image1 = SobelEdgeDetection(GaussianBlur(image1))
    image2 = SobelEdgeDetection(GaussianBlur(image2))
    return HammingDistance(image1, image2)


def GaussianBlur(image):
    return cv2.GaussianBlur(image, (3, 5), 0)


def SobelEdgeDetection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    edges = cv2.threshold(gradient_magnitude, 0.2 *
                          gradient_magnitude.max(), 255, cv2.THRESH_BINARY)[1]

    return edges


def HammingDistance(image1, image2):
    if len(image1) == len(image2) and len(image1[0]) == len(image2[0]):
        difference = 0
        for row1, row2 in zip(image1, image2):
            for pixel1, pixel2 in zip(row1, row2):
                if pixel1 != pixel2:
                    difference += 1
        return difference


if __name__ == "__main__":
    main()
