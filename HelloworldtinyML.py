# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 10:14:04 2022

@author: ASUS
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import math

#%%

sample=1000
seed=100
np.random.seed(seed)
tf.random.set_seed(seed)

#%%
x_values = np.random.uniform(low=0,high=2*math.pi,size=sample)

#%%
y_values=np.sin(x_values)
#%%

y_values += 0.1 * np.random.randn(*y_values.shape)

#%%

plt.plot(x_values,y_values,'b.')

#%%

train_split=int(0.6*sample)
test_split=int(0.2*sample + train_split)

#%%

x_train,x_valid,x_test=np.split(x_values,[train_split,test_split])
y_train,y_valid,y_test=np.split(y_values,[train_split,test_split])

#%%
model=keras.Sequential([
    layers.Dense(units=16,activation='relu',input_shape=[1]),
    layers.Dropout(0.1),
    layers.Dense(16,'relu'),
    layers.Dense(1)])

model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.summary()

#%%
    
history=model.fit(x_train,y_train,validation_data=[x_valid,y_valid],epochs=600,batch_size=16)

#%%

pred=model.predict(x_train)


#%%

plt.plot(x_train,pred,'b.')
plt.plot(x_test,y_test,'r.')

#%%
#converting a model nto a tfflitemodel
converter =tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model=converter.convert()
#%%
#saving the model
open("sine_model.tflite","wb").write(tflite_model)

#%%
#converting a model with quantization...changing floating point bits into 8 bit integers

converter=tf.lite.TFLiteConverter.from_keras_model(model)

#assigning the default optimizer
converter.optimizations=[tf.lite.Optimize.DEFAULT]

#define a function to provide data to the converter maintain accuracy of the model
#providing test data

def generator_function():
    for value in x_test:
        # Each scalar value must be inside of a 2D array that is wrapped in a list
        yield[np.array(value,dtype=np.float32,ndmin=2)]
        

#%%
converter.representative_dataset=generator_function

#%%
tflite_model=converter.convert()

#%%
open('sine_model_quantized.tflite','wb').write(tflite_model)

#%%
#instantiate an intepreter for each model

sine_model=tf.lite.Interpreter('sine_model.tflite')
sine_model_quantized=tf.lite.Interpreter('sine_model_quantized.tflite')

#%%

#allocate memory

sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()
#%%
#Get indexes of the input and output tensors

sine_model_input_index=sine_model.get_input_details()[0]["index"]
sine_model_output_index=sine_model.get_output_details()[0]["index"]

sine_model_quantized_input_index=sine_model_quantized.get_input_details()[0]["index"]
sine_model_quantized_output_index=sine_model_quantized.get_output_details()[0]["index"]

#%%
#place to store the output
sine_model_predictions=[]
sine_model_quantized_predictions=[]

#%%


for x_value in x_test:
    #create a 2D tensor wrapping
    x_value_tensor=tf.convert_to_tensor([[x_value]],dtype=np.float32)
    
    #put in the input
    sine_model.set_tensor(sine_model_input_index,x_value_tensor)
    
    #run the model
    sine_model.invoke()
    
    #read predictions
    sine_model_predictions.append(sine_model.get_tensor(sine_model_output_index)[0])
    
    
    #for the quantised model
    sine_model_quantized.set_tensor(sine_model_quantized_input_index, x_value_tensor)
    
    sine_model_quantized.invoke()
    sine_model_quantized_predictions.append(sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0])   
#%%
epora=range(600)

plt.plot(x_test,sine_model_quantized_predictions,'ro')

plt.plot(x_test,sine_model_predictions,'yo')

#%%








































