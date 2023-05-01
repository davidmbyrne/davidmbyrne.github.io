import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import PIL
import urllib

class SVD:
    def __init__(self):
        pass
    
    def svd_reconstruct(self, image, k):
        def to_greyscale(im):
            return 1 - np.dot(im[...,:3], [0.2989, 0.5870, 0.1140])
        
        grey_img = to_greyscale(image)
        
        U, sigma, V = np.linalg.svd(grey_img)
        
        D = np.zeros_like(grey_img,dtype=float)
        D[:min(grey_img.shape),:min(grey_img.shape)] = np.diag(sigma)
        
        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]
        
        A_ = U_ @ D_ @ V_
        return A_

        
    
        