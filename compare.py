
#Import packages
import matplotlib
matplotlib.use('TkAgg')

import caffe
import lmdb
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.image as mpimg
import sys
import math
import csv
import gc
import os
from scipy.stats import norm
import pickle

import scipy.stats





def pkFeatures(directory_pk,filename,feature_name): 
	path_pk = os.path.join(directory_pk,filename)
	fileObject = open(path_pk,'wb') 
	pickle.dump(feature_name,fileObject)   
	fileObject.close()


def unpkFeatures(directory_pk,filename):
	file_read = os.path.join(directory_pk,filename)
	fileObject = open(file_read,'r')  
	# load the object from the file into var b
	return pickle.load(fileObject)  

#Define distance metric for simialrity
def chiDist(v1,v2):
    d = 0
    epsilon = 0.000001
    if len(v1) != len(v2):
        raise Exception("Both vectors should be of same length")
    else:
        for idx_v in range(0,len(v1)):
            d +=  2*((v1[idx_v] - v2[idx_v])**2)/(v1[idx_v] +v2[idx_v] + epsilon)
    return np.sqrt(d)


def simScore(v1,v2):
    score = 0
    epsilon = 0.001
    threshold = 0.17
    if len(v1) != len(v2):
        raise Exception("Both vectors should be of same length")
    else:
        for idx_v in range(0,len(v1)):
	    if v1[idx_v]== 0 and v2[idx_v]== 0:
            	error = 1
	    else:
		error = abs(v1[idx_v] - v2[idx_v])/(v1[idx_v] +v2[idx_v])

            if  error < threshold :
                score += 1
    return score


def wSimScore(v1,v2):
    w_score = 0
    w_epsilon = 2
    if len(v1) != len(v2):
        raise Exception("Both vectors should be of same length")
    else:
        for idx_v in range(0,len(v1)):
	    if v1[idx_v]== 0 and v2[idx_v]== 0:
            	w_error = 5 
	    else:
                w_error = abs(v1[idx_v] - v2[idx_v])
            
            if  w_error < w_epsilon :
                if abs(np.log(w_error)) == np.inf:
                    w_score += 1
                else:
                    w_score += abs((v1[idx_v] +v2[idx_v])*w_error)
                    
    return w_score





def compare(img1,img2):
    
        gc.collect()
        caffe.set_mode_gpu()
        caffe.set_device(1)


        #model_def = "/home/agupt013/FACES/vgg_face_caffe/VGG_FACE_deploy.prototxt"
        #model_weights = "/home/agupt013/FACES/vgg_face_caffe/VGG_FACE.caffemodel"

        model_def = "/home/agupt013/FACES/python_codes/Styled/deploy.prototxt"
        model_weights = "/home/agupt013/FACES/python_codes/Styled/styled_iter_45000.caffemodel"

        net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

        net_p = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)


        #sys.stdout.write("Starting Validation...")


        # Input Images - Need to confirm how images will be sent RGB BGR, Need Image in BGR array
        val_datum.ParseFromString(value)
        val_label = val_datum.label
        data = caffe.io.datum_to_array(val_datum)

        val_datum_p.ParseFromString(value_p)
        val_label_p = val_datum_p.label
        data_p = caffe.io.datum_to_array(val_datum_p)
        
     
        net.blobs['data'].data[...] = data[:,:,:]
        output = net.forward()
        net_p.blobs['data'].data[...] = data_p[:,:,:]
        output_p = net_p.forward()

                
        val_fc7 = net.blobs['fc7'].data    
        val_fc7_p = net_p.blobs['fc7'].data

        delta_val_fc7 = val_fc7-val_fc7_p
        val_sim = LA.norm(delta_val_fc7, 2)
        
  
        prob_sim = scipy.stats.norm(mean, std).pdf(val_sim)
        prob_dis = scipy.stats.norm(mean1, std1).pdf(val_sim)

        
	sim_score = simScore(val_fc7[0],val_fc7_p[0])

        if val_sim == 0 :
            return 1
        elif prob_sim > 1.1 * prob_dis:
            return 1
        elif prob_dis > 1.1 * prob_sim :
            return 0
        else:
            return 2 
        
                           

        
