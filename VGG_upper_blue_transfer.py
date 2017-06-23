
from __future__ import division

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.utils.visualize_util import plot
from keras.utils import np_utils

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
#keras.backend.image_dim_ordering='th'


from keras import backend as K
import theano
theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

train_index=pd.read_csv('save_targets.csv')


def landmarks(df):
    my_list=[df.landmark_location_x_1,df.landmark_location_y_1,df.landmark_location_x_2,df.landmark_location_y_2,
             df.landmark_location_x_3,df.landmark_location_y_3,df.landmark_location_x_4,df.landmark_location_y_4]
    normed=[i/300 for i in my_list]
    return normed

train_index['lms']=train_index.apply(landmarks,axis=1)
train_index=train_index.set_index('image')

Y1=train_index[['landmark_visibility_1']]
Y2=train_index[['landmark_visibility_2']]
Y3=train_index[['landmark_visibility_3']]
Y4=train_index[['landmark_visibility_4']]

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
vgg16 = VGG16(weights='imagenet')

first = vgg16.get_layer('fc2').output
#prediction = Dense(output_dim=1, activation='sigmoid', name='logit')(first)

first = Dense(1024, activation='relu', name='fc6_pose')(first)
first = Dense(1024, activation='relu', name='fc7_pose')(first)

# predict locations

# output 2 probabilities

vis1 = Dense(2, activation='softmax', name='visibility1')(first)
vis2 = Dense(2, activation='softmax', name='visibility2')(first)
vis3 = Dense(2, activation='softmax', name='visibility3')(first)
vis4 = Dense(2, activation='softmax', name='visibility4')(first)
lm=Dense(8,activation='linear',name='landmarks')(first)

outputs=[vis1, vis2, vis3, vis4,lm]

model = Model(input=vgg16.input, output=outputs)


from keras.preprocessing import image
import keras.backend as K

train_gen= keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

from keras.preprocessing import image
import keras.backend as K

train_gen= keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

# limiting batch size to 16 ~ due to GPU mem issues
train_generator = train_gen.flow_from_directory(
        '/home/icarus/mount_data/lower2/train',
        target_size=(224, 224),
        batch_size=16,
        class_mode='sparse')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        '/home/icarus/mount_data/lower2/test',
        target_size=(224, 224),
        batch_size=16,
        class_mode='sparse')

def get_labels(labels,generator):
    # ex: all filenames of train_generator
    get_list=generator.filenames
    labs=labels.loc[get_list]
    labs=labs.as_matrix()
    labs=np_utils.to_categorical(labs)
    return labs
    

train_lab_1=get_labels(Y1,train_generator)
test_lab_1=get_labels(Y1,test_generator)

train_lab_2=get_labels(Y2,train_generator)
test_lab_2=get_labels(Y2,test_generator)

train_lab_3=get_labels(Y3,train_generator)
test_lab_3=get_labels(Y3,test_generator)

train_lab_4=get_labels(Y4,train_generator)
test_lab_4=get_labels(Y4,test_generator)


landmarks=train_index[['lms']]

def get_labels_lm(labels,generator):
    # ex: all filenames of train_generator
    get_list=generator.filenames
    labs=labels.loc[get_list]
    return np.array(list(labs))

lm_train_labels=get_labels_lm(train_index.lms,train_generator)
lm_test_labels=get_labels_lm(train_index.lms,test_generator)
#lm_train_labels[:5]

# now we have 99608
def train_flow_from_directory(flow_from_directory_gen, list_of_values1,list_of_values2,list_of_values3,list_of_values4,list_of_values5):
    # create infinite loop for labels, 
        
    for i,x in enumerate(flow_from_directory_gen):
        i = i%6225
        # control for half of a batch
        length=len(x[0])
        yield(x[0],[list_of_values1[i*16:i*16+length],list_of_values2[i*16:i*16+length],list_of_values3[i*16:i*16+length],list_of_values4[i*16:i*16+length],list_of_values5[i*16:i*16+length]])
# now we have 19158
def test_flow_from_directory(flow_from_directory_gen, list_of_values1,list_of_values2,list_of_values3,list_of_values4,list_of_values5):
    # create infinite loop for labels, 
        
    for i,x in enumerate(flow_from_directory_gen):
        i = i%1197
        # control for half of a batch
        length=len(x[0])
        # outputs need to be in a list! 
        yield(x[0],[list_of_values1[i*16:i*16+length],list_of_values2[i*16:i*16+length],list_of_values3[i*16:i*16+length],list_of_values4[i*16:i*16+length],list_of_values5[i*16:i*16+length]])
            
#regression_flow_from_directory(train_generator, list_in_order)

def custom_objective(y_true, y_pred):
    '''tailored cost function'''
    
    #get landmarks

    pred_lm=y_pred[4]
    true_lm=y_true[4]
    
    pred_lm=K.reshape(pred_lm,(4,2))
    true_lm=K.reshape(true_lm,(4,2))
    
    
    
    # get visibility binaries
    
    #pred_vis=y_pred[:4]
    true_viz=y_true[:4]
    
    # predict if visible or not, if not visible, set landmark loss to zero
    wins=K.argmax(true_viz,axis=1)
    
    #shapes : wins: (1,4), pred_lm: 4,2
    loss=K.mean(K.square(K.dot(wins,(pred_lm-true_lm))))  # 
    
    
    
    return loss

from keras import backend as K

def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn

# metrics : used to judge the performance (but classifier only makes changes based on )

from keras.optimizers import SGD, Adam

#losses=['sparse_categorical_crossentropy','sparse_categorical_crossentropy','sparse_categorical_crossentropy','sparse_categorical_crossentropy',custom_objective]

losses=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy',custom_objective]
my_metrics=['recall',single_class_accuracy(0)]
#my_metrics=['accuracy','accuracy','accuracy','accuracy',custom_metric]


sgd = SGD( lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=1.0)
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# try adding clipnorm, and clipvalue -- because gradients are going thru da roof

model.compile(loss=losses,optimizer=adam,metrics=my_metrics)

#metrics=my_metrics

filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5'
mcp=keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

cw={0:10,1:1}

callback=[mcp]
test=model.fit_generator(train_flow_from_directory(train_generator, train_lab_1,train_lab_2,train_lab_3,train_lab_4,lm_train_labels),validation_data=test_flow_from_directory(test_generator, test_lab_1,test_lab_2,test_lab_3,test_lab_4,lm_test_labels),
                         nb_epoch=20,class_weight=cw,samples_per_epoch=99000, nb_val_samples=19000,callbacks=callback)
