import cv2
import numpy as np
import glob
import os
from sklearn.utils import shuffle

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    dict_dataset = {}
    f = open('train_labels.txt')
    for line in f:
        l = line.strip().split('|')
        label = int(l[2].strip())
        if label == 5:
            label = 3
	dict_dataset[l[0].strip()] = label

    print ("Done reading dataset dictionary")

    print ("Going to read training images")

    d = {}
    for i,j in enumerate(classes):
        d[j] = i


    for index, fields in enumerate(classes):
        print('Now going to read {} files (Index: {})'.format(fields, index))
        #path = os.path.join(train_path, fields, '*g')
        path = os.path.join(train_path, '*g')
	print "Path", path
        files = glob.glob(path)
	print "Total Training + Val images", len(files)


        for idx, fl in enumerate(files):
	    print "Reading file No", idx
	    fl2 = fl.split('/')[1] 
            label = np.zeros(len(classes))
            #assigning labels
	    
	    label[dict_dataset[fl2]] = 1.0
            
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0/255.0)
            images.append(image)
		


            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            ds = {1:'dry_cans', 0:'dry_other', 4:'dry_paper', 2:'dry_plastic', 3:'wet'}
            cls.append(ds[dict_dataset[fl2]])
        break

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)


    return images, labels, img_names, cls

class DataSet(object):

    def __init__(self, images, labels, img_names, cls):

        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self.img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:

            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

def read_train_sets(train_path, image_size, classes, validation_size):

    class DataSets(object):
        pass
    
    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = (int(validation_size*images.shape[0]))

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]
    
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets


print ("DATASET RAN SUCCESSFULLY")
