# -*- coding: utf-8 -*-
"""
@author: KUZ
"""

import pandas as pd
from pathlib import Path
import ntpath
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam

data_folder = Path('C:/Users/ABCD/Desktop/Self Driving Car/Driving simulator data')
csv_file = data_folder / "driving_log.csv"

columns = ['center','left','right','steering','throttle','reverse','speed']

data = pd.read_csv(csv_file,names=columns)

number_of_bins = 25
hist, bias = np.histogram(data['steering'],bins=number_of_bins)
Center = (bias[:-1] + bias[1:]) * 0.5

#Remove the excess data to balance the remaining data
std_size = 450
indices_to_be_removed = 0
for i in range(len(hist)):
    if(hist[i] > std_size):
        indices_to_be_removed = data[data['steering'] == Center[i]].index
        random.shuffle([indices_to_be_removed])
        data.drop(data.index[indices_to_be_removed[std_size:]],inplace=True)

hist, bias = np.histogram(data['steering'],bins=number_of_bins)
plt.bar(Center,hist,width=0.1)
plt.show()
def load_img_and_steering(data):
    center_image_path = []
    left_image_path = []
    right_image_path = []    
    steering = []
    
    center_image_path = np.array(data['center'])
    left_image_path = np.array(data['left'])
    right_image_path = np.array(data['right'])
    steering = np.array(data['steering'])
        
    return center_image_path,left_image_path,right_image_path,steering

i=0

def preprocess_image(image_path):  
    processed_image = plt.imread(image_path)
    processed_image = processed_image[60:135,:,:]
    processed_image = cv2.cvtColor(processed_image,cv2.COLOR_BGR2YUV)
    processed_image = cv2.GaussianBlur(processed_image,(3,3),0,0)
    processed_image  =cv2.resize(processed_image,(200,66))
    processed_image = processed_image/255
    
    return processed_image  
 
cip,lip,rip,steering = load_img_and_steering(data)
#image = plt.imread(cip[random.randrange(0,len(cip))])
#processesd_image = preprocess_image(image)

X_train, X_test, y_train, y_test = train_test_split(cip,steering,test_size  = 0.2,random_state=3)
y_train_df = pd.DataFrame(y_train)

#def process_all_images():
#    processed_images=[]
#    for i in range(0,len(X_train)):
#        image = plt.imread(X_train[i])
#        processed_image = preprocess_image(image)
#        print(i,len(X_train))
#        processed_images = np.append(processed_images,processed_image)
    
#fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(9,4.5))
#plt.tight_layout()
#plt.grid()

#axes[0].imshow(image)
#axes[0].set_title("original")
#axes[1].imshow(processesd_image)
#axes[1].set_title("processed")
#
#plt.show()
#
#print(X_train.shape)

X_train = np.array(list(map(preprocess_image,X_train)))
X_test = np.array(list(map(preprocess_image,X_test)))

def nvidia_model():
    
    model = Sequential()
    model.add(Conv2D(24,(5,5),subsample=(2,2),input_shape = (66,200,3),activation='relu'))
    model.add(Conv2D(36,(5,5),subsample=(2,2),activation='elu'))
    model.add(Conv2D(48,(5,5),subsample=(2,2),activation='elu'))
    model.add(Conv2D(64,(3,3),activation='elu'))
    model.add(Conv2D(64,(3,3),activation='elu'))
    model.add(Dropout(rate=0.65))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(rate=0.65))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(rate=0.65))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam,loss='mean_squared_error',metrics=['mae', 'acc'])

    return model

model = nvidia_model()
print(model.summary())

history = model.fit(X_train,y_train,epochs=40,batch_size=100,verbose=1,validation_split=0.2,shuffle=True)


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train_loss','val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('LOSS')
plt.show()


model.save("modelneu.h5")
