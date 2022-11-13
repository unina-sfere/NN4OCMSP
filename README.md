NN4OCMSP 
================

This repository contains Python code and data of the paper of Lepore, Palumbo and Sposito 
*Out-of-control signal interpretation in multiple stream processes using artificial neural networks*.

This repository contains the following files:

-   NN4OCMSP/data contains the HVAC data set
-   NN4OCMSP/dataset.py allows the user to access the HVAC data set from the `NN4OCMSP`package
-   NN4OCMSP/functions.py is the source code of the Python package `NN4OCMSP` 
-   NN4OCMSP_tutorial.ipynb is the Jupyter Notebook performing all the analysis shown in
    the Section "*A real-case study*" of the paper

Moreover, in the following Section we provide a tutorial to show how to implement in Python 
the proposed methodology used in the paper to the real-case study.

# Out-of-control signal interpretation in multiple stream processes using artificial neural networks

## Introduction

This tutorial shows how to implement in Python the proposed methodology to the 
real-case study to diagnose faults in the HVAC systems installed on board of passenger railway vehicles. 
The operational data were acquired and made available by the rail transport company Hitachi 
Rail STS based in Italy.
HVAC data set contains the data analyzed in the paper and can be loaded by using the function `load_HVAC_data()`. 
Alternatively, one can use another data set and apply this methodology to any multiple stream process.

You can install the development version of the Python package `NN4MSP` from GitHub with

``` python
pip install git+https://github.com/unina-sfere/NN4OCMSP#egg=NN4OCMSP
```

``` python

# Import libraries

import math 
import pandas as pd
import numpy as np
import sklearn
import keras
from keras import Sequential
from keras.layers import Dense

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

from itertools import combinations
import random as python_random
import tensorflow as tf

from NN4OCMSP.functions import *
import NN4OCMSP.dataset

```
## Real-case study: HVAC systems in passenger railway vehicles 

``` python

# Import HVAC data 

HVAC_data = NN4OCMSP.dataset.load_HVAC_data()

```

``` python

# Filter train A for Phase I estimatin

train_1_data = HVAC_data[HVAC_data["Vehicle"] == "Train_A"]

# Select the DeltaT variables 
train_1_data = train_1_data.iloc[:,-6:]
# Convert pandas dataframe to NumPy array
train_1_data = train_1_data.to_numpy()

```

``` python

# MSP

s = 6 # number of streams corresponding to the number of train coaches
n = 5 # sample size

# Estimate the mean, the process variability and the variability between streams from the Phase I
# data using standard one-way ANOVA techniques 

mu, sigmaA, sigmae = phaseI_estimation(train_1_data)
print('mean', mu)
print('standard deviation of the common stream component', sigmaA)
print('standard deviation of the individual stream component', sigmae)

```

mean 0.08 

standard deviation of the common stream component 0.85 

standard deviation of the individual stream component 0.33 
 
``` python

# Plot the ΔT signals from the six train coaches

train_1_data = train_1_data.transpose().reshape(-1,n).mean(1).reshape(s,-1).transpose() # Average every 5 rows 

# Plot the ΔT signals from the six train coaches

fig = plt.figure(figsize=(12, 8))

x = np.arange(1, 26 ,1)

a = 0
b = 25

plt.plot(x,train_1_data[a:b,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_1_data[a:b,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_1_data[a:b,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_1_data[a:b,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_1_data[a:b,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_1_data[a:b,5], label = 'Coach 6', color='violet', ls='-', marker='P')
# plt.axhline(0, color="red")
plt.xlabel('Sample', fontsize=12)
plt.ylabel('$ \Delta$T', fontsize=12)
plt.legend(fontsize=10)

plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)
plt.xticks(np.arange(1, 26, 1))

plt.show()

```

![](https://github.com/unina-sfere/NN4OCMSP/blob/main/README_Figure/plot_DeltaT_PhaseI_train_A.png)


### Neural Network training

``` python

# Set the simulation parameters to properly generate the data set to train the Neural Network (NN)

num_samples = 10000 # number of samples for each out-of-control (oc) scenario
alpha_sim = 0.05 # Type-I error

# Generate data
X, y = dataset_generator(s,n,num_samples, mu, sigmaA, sigmae,alpha_sim)

```

``` python

# Train/validation split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify = y ,random_state=27)

```

``` python

# Data are normalized to have unit variance and zero mean

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

```

``` python

# NN definition 

num_hidden_layer = 2 # Number of hidden layers
hidden_activation_function = ['relu', 'relu'] # activation function in the hidden layers
number_hidden_neuron = [20,10] # number of neurons in the hidden layers
num_output_neuron = s # The output layer consists of 𝑠 neurons, corresponding to the 𝑠 possible streams 
                      # responsible for the OC signal
    
epochs = 10 # Number of epochs to train the model. An epoch is an iteration over the entire training data
batch_size = 516 # Number of samples per gradient update

# NN Training 

classifier = NN_model(hidden_activation_function, num_hidden_layer,number_hidden_neuron, num_output_neuron) 

# Compiling the neural network

classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy']) # Configures the model for training

# Fitting 

history = classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_val, y_val)) # Trains the model

```

### Train B


``` python

train_2_data = HVAC_data[HVAC_data["Vehicle"] == "Train_B"] # Filter Vehicle by Train B 
train_2_data = train_2_data.iloc[0:-2,-6:] # Select the DeltaTemp variables 
train_2_data = train_2_data.to_numpy() # Convert pandas dataframe to NumPy array
train_2_data = train_2_data.transpose().reshape(-1,n).mean(1).reshape(s,-1).transpose() # Average every 5 rows   

```


``` python

# Plot the ΔT signals from the six train coaches 

fig = plt.figure(figsize=(12, 6))

x = np.arange(1,16,1)

a = 8
b = 23

plt.plot(x,train_2_data[a:b,0], label = 'Coach 1', color='black', ls='-', marker='*')
plt.plot(x,train_2_data[a:b,1], label = 'Coach 2', color='blue', ls='-', marker='.')
plt.plot(x,train_2_data[a:b,2], label = 'Coach 3', color='red', ls='-.', marker= 's')
plt.plot(x,train_2_data[a:b,3], label = 'Coach 4', color='green', ls='-', marker='D')
plt.plot(x,train_2_data[a:b,4], label = 'Coach 5', color='orange', ls='-', marker='+')
plt.plot(x,train_2_data[a:b,5], label = 'Coach 6', color='violet', ls='-', marker='P')
plt.xlabel('Sample', fontsize=12)
plt.ylabel('$ \Delta$T', fontsize=12)
plt.legend(fontsize=10)

plt.xticks(np.arange(1, 16, 1))
plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)

plt.show()

```

![](https://github.com/unina-sfere/NN4OCMSP/blob/main/README_Figure/plot_DeltaT_PhaseII_train_B.png)


``` python

# Compute the overall mean and the range for each sample

overall_mean, sample_range  = range_overall_mean(train_2_data[a:b,])

```


``` python

# Design and plot of the overall mean and range control charts

fig_size = (12, 6)
fig_control_chart = plt.figure(figsize=fig_size)
fig_control_chart= control_charts(fig_control_chart, s, n, mu, sigmaA, sigmae, alpha_sim, overall_mean, sample_range)

```
![](https://github.com/unina-sfere/NN4OCMSP/blob/main/README_Figure/plot_control_charts_train_B.png)

``` python

#  Predict the OC stream(s) at the time of the first signal

pred = prediction(train_2_data[a:b,], classifier, scaler, overall_mean, sample_range)

# The range control chart starts signaling an OC state from sample 9. 
pred[8]

```
array([0, 0, 0, 0, 0, 1])
