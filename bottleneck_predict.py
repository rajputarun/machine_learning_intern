#import
import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
import os

def predict():
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    image_path = 'test/im (67).jpg'

    orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model,use exact same configuration as used for training
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    top_model_weights_path = 'bottleneck_fc_model.h5'
    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)
    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]
    #process the image path to get filename
    f_name = str(image_path).split('/')[-1:][0]
    # get the prediction label and print along with ID
    print("ImageID: {}, Label: {}, Filename: {}".format(inID, label, f_name))
    #save data to csv using pandas dataframe
    d={'Filename':[f_name], 'Label':[label]}
    df = pd.DataFrame(d, columns = ['Filename', 'Label'])
    
# if file does not exist write header 
    if not os.path.isfile('predictions.csv'):
        df.to_csv('predictions.csv', sep='\t', mode='a', header =True, encoding='utf-8', index=False)
    else: # else it exists so append without writing the header
        df.to_csv('predictions.csv', sep='\t', mode = 'a', header=False, encoding='utf-8', index=False)

    
predict()

