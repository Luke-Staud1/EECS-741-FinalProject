import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
from scipy.spatial.distance import hamming

#function to load gallery images. Return list of images
def loadGallery(PATH):
    gallery = [] #array to hold images
    #PATH = '/GallerySet/' #path to gallery images
    for file in os.listdir(PATH): #for every file in the directory
        imgPATH = os.path.join(PATH,file) #get file path
        img = cv2.imread(imgPATH, cv2.IMREAD_GRAYSCALE) #read in images
        gallery.append(img) #append to gallery array
    return np.array(gallery) #return np array of gallery


def imageBinarization(arr, threshold=128):
    #height,width = arr.shape #get height and width of numpy array
    #for r in range(height): #for every row
     #   for c in range(width): #for every column
      #      if (arr[r,c]<=threshold): #if the pixel at location (r,c) is above or equal to threshold, set to 1
       #         arr[r,c] = 255
        #    else: #otherwise set pixel to 0
         #       arr[r,c] = 0
    #return arr #return new image array
    x = arr.shape[0]
    for i in range(x):
        if (arr[i]>threshold):
            arr[i]=255
        else:
            arr[i]=0
    return arr

#normalize the images (used before binarization
def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)

#function to calculate decibility score
#def calcDecidability(scoreMatrix):
    #since the first probe is the same person as the first gallery, second probe
    #same person as second gallery, etc... the diagonal scores are the genuine
    #scores
 #   genuineScore = np.diagonal(scoreMatrix) 


def calculate_decidability_index(score_matrix):
    # Extract the genuine scores from the score matrix
    genuine_scores = np.diagonal(score_matrix)

    # Create a mask with the same shape as the score matrix
    mask = np.ones(score_matrix.shape, dtype=bool)

    # Set the diagonal elements to False
    np.fill_diagonal(mask, False)

    # Extract the imposter scores from the score matrix
    imposter_scores = score_matrix[mask]

    # Calculate the mean of the genuine scores
    genuine_mean = np.mean(genuine_scores)

    # Calculate the mean of the imposter scores
    imposter_mean = np.mean(imposter_scores)

    # Calculate the standard deviation of the genuine scores
    genuine_std = np.std(genuine_scores)

    # Calculate the standard deviation of the imposter scores
    imposter_std = np.std(imposter_scores)

    # Calculate the decidability index (d')
    d_prime = np.abs(genuine_mean - imposter_mean) / np.sqrt(0.5 * (genuine_std**2 + imposter_std**2))
    print(d_prime)
    return d_prime




def main():
    galleryPATH = 'GallerySet/'
    galleryImg = loadGallery(galleryPATH) #get np array of gallery images

    avgFace = np.mean(galleryImg, axis=0) #get the "average face"

    #subtract every image in galleryImg from avgFace
    differenceGallery = galleryImg - avgFace
    
    #flatten all the images
    #flattenGallery = np.vstack([image.reshape(-1) for image in
    #differenceGallery])

    #n = 20 # number of principal components to keep

    #perform principal component analysis
   # pca = PCA(n_components=n, whiten=True)
   # pca.fit(differenceGallery)
    #note: *differenceGallery[0].shape passes the shape of the first image as
    #args
   # eigenfaces = pca.components_.reshape((n, *differenceGallery[0].shape))
    #galleryPrincipalComp is a 2d array, where a row is an image,
    # and every column is one of n principal components
   # galleryPrincipalComp = pca.transform(flattenGallery)
    flattenGallery = np.vstack([image.reshape(-1) for image in differenceGallery])

    n = 99 # number of principal components to keep

    pca = PCA(n_components=n, whiten=True)
    pca.fit(flattenGallery)  # Pass the flattened gallery array here
    eigenfaces = pca.components_.reshape((n, *differenceGallery[0].shape))
    galleryPrincipalComp = pca.transform(flattenGallery)


    #******************Probe Images**********************
    probePATH = 'ProbeSet/'
    probeImg = loadGallery(probePATH) #get np array of probe images
    
    #subtract avgFace from all probe images
    differenceProbe = probeImg - avgFace

    #flatten probe images
    flattenProbe = np.vstack([image.reshape(-1) for image in differenceProbe])
    probePrincipalComp = pca.transform(flattenProbe)

    #normalize the images before binarizing
    normalizedGallery = np.array([normalize(img) for img in galleryPrincipalComp])
    normalizedProbe = np.array([normalize(img) for img in probePrincipalComp])


    # Binarize gallery and probe images
    binarizedGallery = np.array([imageBinarization(img) for img in
    normalizedGallery])
    binarizedProbe = np.array([imageBinarization(img) for img in normalizedProbe])

    #create empty score matrix of size 100x100
    scoreMatrix = np.zeros((100,100))
    #fill score matrix
    for i,probe in enumerate(binarizedProbe):
        for j,gallery in enumerate(binarizedProbe):
            scoreMatrix[i,j] = hamming(probe,gallery)

    np.set_printoptions(edgeitems=100, linewidth=200, suppress=True)
    print(scoreMatrix)

    calculate_decidability_index(scoreMatrix)

    return 0



if __name__ == '__main__':
    main()
