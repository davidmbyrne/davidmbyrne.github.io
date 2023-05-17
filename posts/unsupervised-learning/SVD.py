import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import PIL
import urllib

def svd_reconstruct(image, k):
    U, sigma, V = np.linalg.svd(image)
        
    D = np.zeros_like(image, dtype=float)
    D[:min(image.shape),:min(image.shape)] = np.diag(sigma)
        
    U_ = U[:,:k]
    D_ = D[:k, :k]
    V_ = V[:k, :]
        
    A_ = U_ @ D_ @ V_
    return A_

def read_image(url):
    return np.array(PIL.Image.open(urllib.request.urlopen(url)))

def to_greyscale(im):
    return 1 - np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])
    
def compare_images(A, A_):

    fig, axarr = plt.subplots(1, 2, figsize = (15, 10))

    axarr[0].imshow(A, cmap = "Greys")
    axarr[0].axis("off")
    axarr[0].set(title = "original image")

    axarr[1].imshow(A_, cmap = "Greys")
    axarr[1].axis("off")
    axarr[1].set(title = "reconstructed image")
    
    
def svd_experiment(img):
    
    fig, axarr = plt.subplots(3, 2, figsize = (15, 10))
    m, n = img.shape
    
    axarr[0, 0].imshow(svd_reconstruct(img, 10), cmap = "Greys")
    axarr[0, 0].axis("off")
    axarr[0, 0].set(title = str(10) + " Components, " + str(round((10*m+10*n)/(m*n)*100, 2)) + "% storage")
    
    axarr[0, 1].imshow(svd_reconstruct(img, 20), cmap = "Greys")
    axarr[0, 1].axis("off")
    axarr[0, 1].set(title = str(20) + " Components, " + str(round((20*m+20*n)/(m*n)*100, 2)) + "% storage")
    
    axarr[1, 0].imshow(svd_reconstruct(img, 30), cmap = "Greys")
    axarr[1, 0].axis("off")
    axarr[1, 0].set(title = str(30) + " Components, " + str(round((30*m+30*n)/(m*n)*100, 2)) + "% storage")
    
    axarr[1, 1].imshow(svd_reconstruct(img, 40), cmap = "Greys")
    axarr[1, 1].axis("off")
    axarr[1, 1].set(title = str(40) + " Components, " + str(round((40*m+40*n)/(m*n)*100, 2)) + "% storage")
    
    axarr[2, 0].imshow(svd_reconstruct(img, 50), cmap = "Greys")
    axarr[2, 0].axis("off")
    axarr[2, 0].set(title = str(50) + " Components, " + str(round((50*m+50*n)/(m*n)*100, 2)) + "% storage")
    
    axarr[2, 1].imshow(svd_reconstruct(img, 60), cmap = "Greys")
    axarr[2, 1].axis("off")
    axarr[2, 1].set(title = str(60) + " Components, " + str(round((60*m+60*n)/(m*n)*100, 2)) + "% storage")
    