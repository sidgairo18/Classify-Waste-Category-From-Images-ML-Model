import os
import pdb
import cv2
import numpy as np

genres = ['action','adventure', 'animation', 'biography', 'comedy', 'crime' , 'documentary' , 'drama' , 'family', 'fantasy', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'short', 'sport', 'thriller', 'war', 'western']


ct = 0
for genre in genres:

    l = os.listdir(genre)

    for i, name in enumerate(l):
        #print ('./'+genre+'/'+name)
        im = cv2.imread('./'+genre+'/'+name)
        if im != None:
            ct += 1

print (ct)
