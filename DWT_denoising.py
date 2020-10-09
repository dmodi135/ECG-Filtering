#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:33:56 2020
@author: miagiandinoto
(and raymond)

data from https://physionet.org/content/stdb/1.0.0/
"""
# this function takes in inputs noisySignal(the raw ECG), channels, beginning point of sample,
# and endpoint of sample.
# outputs the denoised signal, as well as the array of coefficients used for DWT
# also plots denoised signal

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import wfdb
import pywt
from skimage.restoration import denoise_wavelet

def DWT_denoising(noisySignalFile,channels, sampfrom, sampto):

    noisySignal1, fields = wfdb.rdsamp(noisySignalFile, channels=[channels],sampfrom=sampfrom, sampto=sampto)
    noisySignal=noisySignal1.flatten()
    N = fields.get('sig_len')
    fs = fields.get('fs')

    tfrom = sampfrom / fs
    tto = sampto / fs
    times = np.linspace(tfrom, tto, N)

    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(2,1,1)
    plt.plot(times, noisySignal, linewidth= 0.5)

    level = 8
    waveName = 'db6'
    coeffs = pywt.wavedec(noisySignal, waveName, mode='symmetric', level=level)
    
   
    cD1 = np.array(coeffs[level])
    # THRESHOLD SELECTION
    mediancD1 = np.median(cD1)
    sigma = mediancD1 / .6457
    t = sigma * math.sqrt(2 * math.log(N, 10))

    newCoeffs = [(pywt.threshold(subArr,t,mode='hard')) for subArr in coeffs]


    denoised = pywt.waverec(newCoeffs, waveName, mode='symmetric')
    
    # denoised = denoise_wavelet(noisySignal, method='BayesShrink', wavelet_levels=level, wavelet=waveName, mode='soft')
    fig.add_subplot(2,1,2)
    plt.plot(denoised, linewidth=0.5,color='red')
    #plt.savefig(os.getcwd() + '/mia_wavelet_filter')
    plt.show()

    return(denoised,coeffs,newCoeffs)

os.chdir('/Users/dhruvmodi/Desktop/ECG-Filtering/ecg-database/Person_01/')
denoised,coeffs,newCoeffs=DWT_denoising('rec_1',0, 0, 1000)


