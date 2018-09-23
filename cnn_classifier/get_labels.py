import os
import pdb
import cv2
import numpy as np

genres = ['action','adventure', 'animation', 'biography', 'comedy', 'crime' , 'documentary' , 'drama' , 'family', 'fantasy', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'short', 'sport', 'thriller', 'war', 'western']

f = open('corrected.txt')

d = {}


for line in f:
    l = line.strip().split('|')
    d[l[4].strip()] = l[3].strip()
    #print (l[4].strip(), l)

#print ("Dictionary Got")
f.close()

ct = 0
for genre in genres:

    l2 = os.listdir(genre)
    d2 = {}

    f2 = open('./'+genre+'.txt')

    for line in f2:
        l = line.strip().split('#')
        #print (l[1], l[-2])

        d2[l[1].strip()+'.jpg'] = l[-2].strip()

    f2.close()

    for i, name in enumerate(l2):
        print ('./'+genre+'/'+name)
        im = cv2.imread('./'+genre+'/'+name)
        if im != None:
            #print ('./'+genre+'/'+name)
            f = open('./'+genre+'/'+name+'.txt','w')
            try:
               cats = d2[name].split(',')
               print ("cats found", cats)
            except:
               ct += 1
               cats = [genre]
            for cat in cats:
                f.write(cat.strip()+'\n')

print (ct)
