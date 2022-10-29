# Import packages

import math
import random as python_random
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

# Functions

def dataset_generator(s, n, num_samples, mu, sigmaA, sigmae, alpha_sim):
    
    """ Return the simulated data set, as described in the reference paper, to 
        train and design the Neural Network for multi-label classification.

    # Parameters
        s: int, number of streams.
        n: int, subgroup size.
        num_samples: int, number of samples for each simulated OC scenario
        mu: int, process mean   
        sigmaA_std: int, standard deviation of the common stream component 𝐴t
        sigmaE_std: int, standard deviation of the individual stream component 
            etjk
        alpha_sim: int, Type-I error of the range and the overall mean control
            chart
            
    # Returns
        X: ndarray, input data    
        y: ndarray, label 
        
    """

    # MSP control charts    
    num_neg_samples = 100000
    np.random.seed(10)
    negative_samples = np.random.normal(loc = mu, scale = sigmae, size = (num_neg_samples *n ,s))
    np.random.seed(11)
    negative_samples = negative_samples + np.random.normal(loc = mu, scale = sigmaA, size = (num_neg_samples * n,1))
    negative_samples = negative_samples.transpose().reshape(-1,n).mean(1).reshape(s,-1).transpose()
    sample_range = negative_samples.max(axis=1) - negative_samples.min(axis=1) 
    UCL_range = np.quantile(sample_range, 1 - alpha_sim)
    sample_average = negative_samples.mean(axis=1)     
    UCL_average = np.quantile(sample_average, 1 - alpha_sim/2)
    
    shift = [-1,-2,-3,1,2,3] 
    positive_samples_all = np.zeros((1, s+2))
    positive_label_all = np.zeros((1, s))
    positive_label = np.zeros((1, s)) 
    
    for i in shift:
        for j in range(s):  # -1
            for l in combinations([i for i in range(s)], j + 1):
                count = 0
                positive_samples_scenario = np.zeros((1, s+2))
                while(count < num_samples):
                    np.random.seed(1 + count) # seed 
                    positive_samples = np.random.normal(loc = mu, scale = sigmae, size = (num_samples*n,s))
                    np.random.seed(1 + count + 1)
                    positive_samples = positive_samples + np.random.normal(loc = mu, scale = sigmaA, size = (num_samples*n,1))
                    positive_samples[:, np.array(l)] = positive_samples[:, np.array(l)] + i*sigmae
                    positive_samples = positive_samples.transpose().reshape(-1,n).mean(1).reshape(s,-1).transpose()
                    
                    overall_mean = positive_samples.mean(axis=1) 
                    sample_range = positive_samples.max(axis=1) - positive_samples.min(axis=1) 
                    positive_samples = np.c_[positive_samples,overall_mean,sample_range] 
                    mask = np.where((sample_range > UCL_range) | (np.abs(overall_mean) > UCL_average))[0]
                    positive_samples = positive_samples[mask,:]
                    positive_samples_scenario = np.vstack([positive_samples_scenario, positive_samples])
                    count = count + len(mask)
                positive_samples_all = np.vstack([positive_samples_all, positive_samples_scenario[1:(num_samples+1),:]])                
                positive_label = np.zeros((num_samples, s))
                positive_label[:,np.array(l)] = 1 
                positive_label_all = np.vstack([positive_label_all, positive_label])
        
    X = np.delete(positive_samples_all, (0), axis=0)
    y = np.delete(positive_label_all, (0), axis=0)
    
    return X,y


def NN_model(hidden_activation_function, num_hidden_layer, num_hidden_neuron, num_output_neuron):
    """ Return an istance of the Multilayer Perceptron (MLP) classifier.

    # Parameters
        hidden_activation_function: list, activation functions for the hidden layers.
        num_hidden_layer: int, number of hidden layers in the model.
        num_hidden_neuron: list, number of neurons in  the hidden layers.
        num_output_neuron_ int, number of neurons in the output layer. 

    # Returns
        classifier: A MLP model.
    """
    
    # To obtain reproducible result susing Keras during development
    np.random.seed(2)
    python_random.seed(123) 
    tf.random.set_seed(1234) 
    
    output_activation = 'sigmoid'
    output_units = num_output_neuron
    
    classifier = Sequential()
    
    for i in range(num_hidden_layer):
        classifier.add(Dense(units=num_hidden_neuron[i], 
                             activation=hidden_activation_function[i]))
    
    classifier.add(Dense(units=output_units, activation=output_activation))

    return classifier


def accuracy(y_true, y_pred):
    '''
    Compute the accuracy for a multi-label classification
     
    # Parameters
        y_true: 1d array-like, or label indicator array / sparse matrix,
            Ground truth (correct) labels.
        y_pred: 1d array-like, or label indicator array / sparse matrix,
            Predicted labels, as returned by a classifier.
    
     # Returns
        accuracy          
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)*100


def range_overall_mean(data):
    '''
    Compute the range and the overall mean for each sample
     
    # Parameters
        data: array_like, array containing numbers whose range and mean
        are desired.
        
     # Returns
        overall_mean: ndarray, returns a new array containing the mean values
        sample_range: ndarray, returns a new array containing the range values
    
    '''    
    
    overall_mean = np.mean(data, axis= 1)
    sample_range = np.max(data, axis= 1) - np.min(data, axis= 1)
    
    return overall_mean, sample_range 


def control_charts(fig_control_chart, s, n, mu, sigmaA, sigmae, alpha_sim, overall_mean,
                   sample_range):

    """  
    Plot the range and the overall mean control charts 
    # Parameters
        overall_mean: array, overall mean for each sample. 
        sample_range: array, sample range.
        fig_control_chart: a Figure instance.
        xlabel: str, a title for the x axis.
        ylabel: str, a title for the y axis.
        s: int, number of streams.
        n: int, subgroup size.
        mu: int, process mean   
        sigmaA_std: int, standard deviation of the common stream component 𝐴t
        sigmaE_std: int, standard deviation of the individual stream component 
            etjk
        alpha_sim: int, Type-I error of the range and the overall mean control
            chart

    # Returns
        fig_control_chart: a matplotlib.figure.Figure object.  
    """    
        
    num_neg_samples = 100000
    np.random.seed(10)
    negative_samples = np.random.normal(loc = mu, scale = sigmae, size = (num_neg_samples *n ,s))
    np.random.seed(11)
    negative_samples = negative_samples + np.random.normal(loc = mu, scale = sigmaA, size = (num_neg_samples * n,1))
    negative_samples = negative_samples.transpose().reshape(-1,n).mean(1).reshape(s,-1).transpose()
    sample_rangeIC = negative_samples.max(axis=1) - negative_samples.min(axis=1) 
    alpha_sim = alpha_sim
    UCL_range = np.quantile(sample_rangeIC, 1 - alpha_sim)
    sample_average = negative_samples.mean(axis=1)     
    UCL_average = np.quantile(sample_average, 1 - alpha_sim/2)
    
    dim = overall_mean.shape[0]
        
    # f = plt.figure(figsize=(12,6))
    ax1 = fig_control_chart.add_subplot(121)
    x = np.arange(1,dim +1,1)
    ax1.plot(x, overall_mean, color='black', ls='-', marker='o')
    ax1.axhline(UCL_average, color="red", label = "UCL")
    ax1.axhline(-UCL_average, color="red", label = "LCL")
    plt.xlabel('Subgroup', fontsize=12)
    plt.ylabel('Overall mean', fontsize=12)    
    plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)
    plt.xticks(np.arange(1, dim + 1, 1))
    
    ax2 = fig_control_chart.add_subplot(122)
    x = np.arange(1,dim +1,1)
    ax2.plot(x, sample_range, color='black', ls='-', marker='o')
    ax2.axhline(UCL_range, color="red", label = "UCL")
    plt.xlabel('Subgroup', fontsize=12)
    plt.ylabel('Range', fontsize=12)
    plt.tick_params(axis='both', which='major', size = 7, width = 1 , direction = 'out', labelsize = 10)
    plt.xticks(np.arange(1, dim +1, 1))
    
    return fig_control_chart


def prediction(data, classifier, scaler, overall_mean, sample_range):
    """
   Return the prediction vector associated to each input
    # Parameters
        data: array_like, array containing numbers whose predictions are desired
        classifier: tensorflow.python.keras.engine.sequential.Sequential object,
            trained Multilayer Perceptron classifier model.
        scaler: sklearn.preprocessing._data.StandardScaler object, standardize 
            features by removing the mean and scaling to unit variance.
        overall_mean: array, overall mean for each sample. 
        sample_range: array, sample range.
        
    # Returns
       data_pred: array, generates output predictions for the input samples
    """ 
        
    data = np.c_[data,overall_mean,sample_range]
    data_std = scaler.transform(data)
    data_pred = classifier.predict(data_std)
    data_pred = (data_pred > 0.5) * 1
    
    return data_pred
    
    
def phaseI_estimation(data):
    """
    Compute the mean, the process variability and the variability between streams
    using standard one-way ANOVA techniques45
    
    # Parameters
        data: array_like, array containing numbers whose predictions are desired,
        
    # Returns
        mu: int, process mean 
        sigmae: int, standard deviation of the common stream component
        sigmaA: int, standard deviation of the individual stream component
    
        
    """     
    
    mu = np.round(np.mean(data), decimals = 2) 
    
    sum_sigmae = 0

    for i in range(data.shape[0]):
        mean_row = np.mean(data[i,:])
        for j in range(data.shape[1]):
            sum_sigmae = sum_sigmae + (data[i,j] - mean_row)**2
            
    sigmae2 =  sum_sigmae/(data.shape[0]*(data.shape[1]-1))       
    sigmae = np.round(math.sqrt(sigmae2), decimals = 2)

    
    sum_sigmaA = 0

    overall_mean = np.mean(data)
    
    for i in range(data.shape[0]):
        mean_row = np.mean(data[i,:])
        sum_sigmaA = sum_sigmaA + (mean_row - overall_mean)**2
            
    sigmaeA2 =  sum_sigmaA/(data.shape[0] - 1) - sigmae2/data.shape[1]  
    sigmaA = np.round(math.sqrt(sigmaeA2), decimals = 2)

    return mu, sigmaA, sigmae
    



