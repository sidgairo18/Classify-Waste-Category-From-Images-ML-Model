import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import pdb
import os


# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1] 
#filename = dir_path +'/' +image_path

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('my-model.meta')
# Step-2: Now let's load the weights saved using the restore method.


def test(test_dir):
    
    classes = ['dry_other', 'dry_cans', 'dry_plastic', 'dry_paper', 'wet']
    ds = {1:'dry_cans', 0:'dry_other', 4:'dry_paper', 2:'dry_plastic', 3:'wet'} 
    files = os.listdir(test_dir)
    dict_dataset = {}
    image_size = 128


    d = {}
    for i, j in enumerate(classes):
        d[j] = i
    print "Total test Images:", len(files)    
    print "Now going to read the test images"
    
    test_data = []


    print "Reading files"
    im_names = []
    for index, name in enumerate(files):
        
        im_names.append(name)
        image = cv2.imread(test_dir+'/'+name)
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0/255.0)

	test_data.append(image)

    print "Files after Reading", len(test_data)
  
    pred_labs = []
    true = []

    for idx, tup in enumerate(test_data):
	print "Testing id", idx
        true.append(tup)
        output = tester(tup)
        pred_labs.append(np.argmax(output[0]))

    f = open('test_output_cnn.csv', 'w')                                 
                                                                        
    for i in range(len(pred_labs)):                                        
        label = pred_labs[i]                                               
        s = im_names[i]+","+ds[label]+"\n"                         
        f.write(s)                                                      
    f.close() 


    



def tester(image):

    image_size=128
    num_channels=3
    images = []
    # Reading the image using OpenCV
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 5))


    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}

    #result=sess.run(tf.round(y_pred), feed_dict=feed_dict_testing)
    result2=sess.run(y_pred, feed_dict=feed_dict_testing)

    return result2

if __name__ == "__main__":
    
    test_dir = "./testing_data"
    test(test_dir)

    exit(-1)
        
