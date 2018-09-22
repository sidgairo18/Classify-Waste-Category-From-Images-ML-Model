import pdb
import os

l = os.listdir('./')
f = open('train_labels.txt','w' )

for i, d in enumerate(l):
    try:
        images = os.listdir('./'+d)

        for im in images:
            s = im+" | "+d+" | "+str(i)+"\n"
            f.write(s)
    except:
        print("pass", d)
        pass
print("Labels written")
