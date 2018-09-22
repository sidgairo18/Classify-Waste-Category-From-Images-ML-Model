import os
import numpy as np
import pickle
import pdb

x = np.loadtxt(open('click_fc7_features_vgg16.txt'))
names = {}

f = open('click_image_name_list.txt')

for i, name in enumerate(f):
    name = name.strip().split('.')
    name = name[0]
    print name
    names[name] = x[i,:]

fp = open('./click_image_embed.pkl', 'w')
pickle.dump(names, fp)
